# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
from copy import deepcopy

import torch.distributed as dist
import torch.utils.checkpoint
import torch.nn as nn
import transformers

from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_internlm2 import InternLM2ForCausalLM
from .modeling_holistic_embedding import (HolisticEmbedding,
                                        HolisticEmbeddingConfig)

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    # main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, embedding_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.embedding_config.image_size
        patch_size = config.embedding_config.patch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.use_thumbnail = config.use_thumbnail

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = HolisticEmbedding(config.embedding_config)

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.embedding_model = get_peft_model(self.embedding_model, lora_config)
        self.embedding_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            input_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
            query = None,
            hd_input_ids = None,
            hd_attention_mask = None,
            hd_position_ids = None,
            hd_input_embeds = None,
            hd_labels = None,
            hd_loss_weight = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_embeds is None:
            if image_flags is not None:
                #image_flags = image_flags.squeeze(-1)
                pixel_values = pixel_values[image_flags == 1]
            if pixel_values==[]:
                pixel_values = None
            if getattr(self.embedding_model.config, 'pixel_shuffle_loc', None) in ['post']:
                assert hd_input_ids is not None, 'hd_input_ids is required for pixel_shuffle_loc=post'
                embedding_input_ids = hd_input_ids
                embedding_attention_mask = hd_attention_mask
                embedding_position_ids = hd_position_ids
            else:
                embedding_input_ids = input_ids
                embedding_attention_mask = attention_mask
                embedding_position_ids = position_ids
            image_embeds, input_embeds, next_past_key_values, layers_output = self.embedding_model(input_ids=embedding_input_ids,
                                                                                    pixel_values=pixel_values,
                                                                                    attention_mask=embedding_attention_mask,
                                                                                    position_ids=embedding_position_ids,
                                                                                    use_cache=use_cache,)
            """image_embeds = embed_output.last_hidden_state
            input_embeds = embed_output.hidden_states
            next_past_key_values = embed_output.past_key_values
            layers_output = embed_output.attentions"""

            B, N = embedding_input_ids.shape
            image_batch_size = pixel_values.shape[0] if pixel_values is not None else 0
            C = image_embeds.shape[-1]
            input_embeds = input_embeds.reshape(B * N, C)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                #print(f'dynamic ViT batch size: {image_batch_size}, images per sample: {image_batch_size / B}, dynamic token length: {N}')
                if statistics is not None:
                    num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                    self.num_samples += num_samples
                    print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

            if image_batch_size != 0:
                if getattr(self.embedding_model.config, 'pixel_shuffle_loc', None) == 'post':
                    B, N = input_ids.shape
                    llm_input_embeds = torch.zeros(input_ids.shape[1], C, device=input_ids.device, dtype=input_embeds.dtype)
                    llm_selected = input_ids.flatten() == self.img_context_token_id
                    hd_llm_selected = hd_input_ids.flatten() == self.img_context_token_id
                    llm_input_embeds[~llm_selected] = input_embeds[~hd_llm_selected]
                    llm_input_embeds[llm_selected] = image_embeds.reshape(-1, C)
                    input_embeds = llm_input_embeds

            input_embeds = input_embeds.reshape(B, N, C)
        
        else:
            next_past_key_values = []
            if getattr(self.embedding_model.config, 'pixel_shuffle_loc', None) in ['post']:
                embedding_input_embeds = hd_input_embeds
                embedding_attention_mask = hd_attention_mask
                embedding_position_ids = hd_position_ids
            else:
                embedding_input_embeds = input_embeds
                embedding_attention_mask = attention_mask
                embedding_position_ids = position_ids
            for layer_idx, layer_module in enumerate(self.embedding_model.encoder):
                outputs = layer_module(
                    hidden_states=embedding_input_embeds,
                    attention_mask=embedding_attention_mask,
                    position_ids=embedding_position_ids,
                    past_key_value=past_key_values[layer_idx],
                    use_cache=use_cache,
                )
                embedding_input_embeds = outputs[0]
                if use_cache:
                    next_past_key_values.append(outputs[1])

            input_embeds = embedding_input_embeds

        if self.config.normalize_encoder_output:
            input_embeds = input_embeds / input_embeds.norm(dim=-1, keepdim=True)
        
        llm_attention_mask = attention_mask
        llm_position_ids = position_ids

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=llm_attention_mask,
            position_ids=llm_position_ids,
            past_key_values=past_key_values[layer_idx+1:] if past_key_values is not None else None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if use_cache:
            for past_key_value in outputs.past_key_values:
                next_past_key_values.append(past_key_value)
        else:
            next_past_key_values = None
            
        try:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=next_past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=layers_output + outputs.attentions,
            )
        except Exception as e:
            # print(e)
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=next_past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        hd_query = deepcopy(query)
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            hd_image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * int(self.num_image_token // self.downsample_ratio**2) * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            hd_query = hd_query.replace('<image>', hd_image_tokens, 1)
            # print(hd_query)

        model_inputs = tokenizer(query, return_tensors='pt')
        hd_model_inputs = tokenizer(hd_query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        hd_input_ids = hd_model_inputs['input_ids'].cuda()
        hd_attention_mask = hd_model_inputs['attention_mask'].cuda()

        generation_config['eos_token_id'] = eos_token_id
        generation_output = super().generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            hd_input_ids=hd_input_ids,
            hd_attention_mask=hd_attention_mask,
            **generation_config
        )
        generation_output = generation_output[:, input_ids.shape[1]:]

        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, input_embeds=None, 
            tile_pos_offsets=None, hd_input_ids=None, hd_attention_mask=None, img_mask=None, **kwargs
    ):
        
        past_key_values = None
        if past_key_values is not None:
            past_length = past_key_values[-1][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            input_embeds = self.embedding_model.get_input_embeddings(input_ids)
            hd_input_ids = input_ids
            hd_input_embeds = input_embeds

        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        hd_position_ids = kwargs.get('hd_position_ids', None)
        if hd_attention_mask is not None and hd_position_ids is None:
            # create position_ids on the fly for batch generation
            hd_position_ids = hd_attention_mask.long().cumsum(-1) - 1
            hd_position_ids.masked_fill_(hd_attention_mask == 0, 1)
            if past_key_values:
                hd_position_ids = hd_position_ids[:, -hd_input_ids.shape[1]:]

        if input_embeds is not None:
            model_inputs = {'input_embeds': input_embeds, 'hd_input_embeds': hd_input_embeds}
        else:
            model_inputs = {'input_ids': input_ids, 'pixel_values': kwargs.get('pixel_values'), 'hd_input_ids': hd_input_ids}

        model_inputs.update(
            {
                'position_ids': position_ids,
                'past_key_values': past_key_values,
                'use_cache': kwargs.get('use_cache'),
                'attention_mask': attention_mask,
                'hd_position_ids': hd_position_ids,
                'hd_attention_mask': hd_attention_mask,
            }
        )
        return model_inputs