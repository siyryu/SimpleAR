#!/usr/bin/env python
# verify_token.py
# ------------------------------------------------------------
# 验证 extract_token_from_datasets.py 生成的 .npy 文件的正确性
# 输入一个 .npy 文件，使用同样的 tokenizer 把他变回图片并可视化
# ------------------------------------------------------------
"""
验证 extract_token_from_datasets.py 生成的 .npy 文件的正确性。

此脚本:
1. 加载指定的 .npy 文件中的 tokens
2. 初始化与 extract_token_from_datasets.py 相同的 tokenizer
3. 使用 tokenizer 将 tokens 解码回图像
4. 可视化重建的图像并可选择保存

使用示例:
    # 基本用法 - 显示图像
    python verify_token.py /path/to/tokens/12345.npy --vq_model_ckpt /path/to/Cosmos-1.0-Tokenizer-DV8x16x16

    # 保存重建的图像
    python verify_token.py /path/to/tokens/12345.npy --output reconstructed.png

    # 在 CPU 上运行
    python verify_token.py /path/to/tokens/12345.npy --device cpu

    # 不显示图像，只保存
    python verify_token.py /path/to/tokens/12345.npy --output reconstructed.png --no_display
"""
import os
import argparse
import numpy as np
import torch
from torchvision.utils import save_image


from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import (
    CausalVideoTokenizer as CosmosTokenizer,
)


def main(args):
    """
    验证 extract_token_from_datasets.py 生成的 .npy 文件的正确性。

    此函数:
    1. 加载指定的 .npy 文件中的 tokens
    2. 初始化与 extract_token_from_datasets.py 相同的 tokenizer
    3. 使用 tokenizer 将 tokens 解码回图像
    4. 可视化重建的图像并可选择保存

    Args:
        args: 命令行参数，包含:
            - npy_file: .npy 文件路径
            - vq_model_ckpt: tokenizer 模型路径
            - output: 输出图像保存路径 (可选)
            - device: 运行设备 (cuda 或 cpu)
            - no_display: 是否不显示图像
    """
    # 检查输入文件是否存在
    if not os.path.exists(args.npy_file):
        raise FileNotFoundError(f"Token file not found: {args.npy_file}")

    # 检查 tokenizer 路径是否存在
    decoder_path = f"{args.vq_model_ckpt}/decoder.jit"
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder model not found: {decoder_path}")

    # 加载 token 数据
    print(f"Loading tokens from {args.npy_file}...")
    try:
        tokens = np.load(args.npy_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokens: {e}")

    print(f"Loaded tokens with shape: {tokens.shape}")

    # tokens_tensor = torch.from_numpy(tokens).to(torch.bfloat16).to(device=args.device)
    tokens_tensor = torch.from_numpy(tokens).to(device=args.device)

    # 初始化 Tokenizer (与 extract_token_from_datasets.py 保持一致)
    print(f"Initializing tokenizer from {args.vq_model_ckpt}...")
    tokenizer_cfg = TokenizerConfigs["DV"].value
    tokenizer_cfg.update(dict(spatial_compression=16, temporal_compression=8))

    try:
        model = CosmosTokenizer(
            checkpoint_enc=f"{args.vq_model_ckpt}/encoder.jit",
            checkpoint_dec=f"{args.vq_model_ckpt}/decoder.jit",
            tokenizer_config=tokenizer_cfg,
        ).eval().requires_grad_(False)

        # 将模型移至指定设备
        model.to(args.device)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize tokenizer: {e}")

    # 解码 token 为图像
    print("Decoding tokens to image...")
    try:
        with torch.no_grad():
            reconstructed_tensor = model.decode(tokens_tensor)

        print(f"Decoded image tensor shape: {reconstructed_tensor.shape}")

        reconstructed_tensor = reconstructed_tensor.squeeze(2)
        save_image(reconstructed_tensor, args.output, normalize=True, value_range=(-1, 1))

    except Exception as e:
        print(f"Error during decoding or visualization: {e}")
        raise

    print("✅ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="验证 extract_token_from_datasets.py 生成的 .npy 文件的正确性，"
                    "通过将 tokens 解码回图像并可视化。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "npy_file", 
        type=str, 
        help="包含 tokens 的 .npy 文件路径 (由 extract_token_from_datasets.py 生成)"
    )

    parser.add_argument(
        "--vq_model_ckpt", 
        type=str, 
        default="/path_to_tokenizer/Cosmos-1.0-Tokenizer-DV8x16x16",
        help="Tokenizer 模型路径，需要包含 decoder.jit 文件"
    )

    parser.add_argument(
        "--output", 
        type=str, 
        default="", 
        help="重建图像的保存路径，如果不指定则不保存"
    )

    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        choices=["cuda", "cpu"],
        help="运行模型的设备"
    )

    parser.add_argument(
        "--no_display", 
        action="store_true", 
        help="不显示图像 (适用于无头环境)"
    )

    args = parser.parse_args()
    main(args)
