#!/usr/bin/env python3
"""
诊断 torch MPS 状态脚本
模拟 AstrBot 主进程加载后的状态
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("Torch MPS 诊断")
print("=" * 60)

# 模拟 AstrBot 主进程已经导入了 torch
print("\n[1] 模拟 AstrBot 主进程导入 torch...")
import torch
print(f"  torch 版本: {torch.__version__}")
print(f"  MPS built: {torch.backends.mps.is_built()}")
print(f"  MPS available: {torch.backends.mps.is_available()}")

# 检查当前的 MPS 设备
if torch.backends.mps.is_available():
    print(f"  MPS device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    try:
        device = torch.device("mps")
        print(f"  MPS device: {device}")
        # 尝试创建一个简单的 tensor
        x = torch.randn(3, 3, device=device)
        print(f"  ✅ MPS tensor 创建成功: {x.shape}")
    except Exception as e:
        print(f"  ❌ MPS tensor 创建失败: {e}")
else:
    print("  MPS 不可用，使用 CPU")

# 尝试在 torch 导入后禁用 MPS
print("\n[2] 尝试在 torch 导入后禁用 MPS...")
try:
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    print(f"  修改后 MPS available: {torch.backends.mps.is_available()}")
    print(f"  修改后 MPS built: {torch.backends.mps.is_built()}")
except Exception as e:
    print(f"  修改失败: {e}")

# 设置环境变量（模拟我们的配置代码）
print("\n[3] 设置环境变量...")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print(f"  CUDA_VISIBLE_DEVICES: '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
print(f"  PYTORCH_ENABLE_MPS_FALLBACK: '{os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}'")

# 尝试创建 MPS tensor（验证是否真的被禁用）
print("\n[4] 再次尝试创建 MPS tensor...")
try:
    device = torch.device("mps")
    x = torch.randn(3, 3, device=device)
    print(f"  ❌ MPS tensor 创建成功（禁用无效）: {x.shape}")
except Exception as e:
    print(f"  ✅ MPS 被成功禁用: {e}")

print("\n" + "=" * 60)
print("结论：如果 [4] 显示 'MPS tensor 创建成功'，说明 monkey patch 无效")
print("=" * 60)
