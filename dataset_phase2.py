"""
dataset_phase2.py
=================

Phase 2 专用数据集（与 Phase 1 完全隔离）

数据读取、配对、变换、打包成 batch。

核心特点：
✅ 使用真实的混合退化数据（input/label 配对）
✅ 不使用任何人工增强
✅ 支持训练/验证/测试模式
✅ 与 Phase 1 的 dataset.py 完全独立

数据要求：
- input: 真实水下退化图像（包含雾、色偏、暗光等混合退化）
- label: 对应的干净图像

目录结构：
    dataset/
    ├── train/
    │   ├── input/   (真实混合退化图像)
    │   └── label/   (ground truth)
    ├── val/
    │   ├── input/
    │   └── label/
    └── test/
        ├── input/
        └── label/

1. 读取 input_dir 中的退化图像
2. 读取 label_dir 中的清晰 GT 图像
3. 检查 input 和 label 数量是否一致
4. 检查文件名是否一一对应
5. PIL 读取 RGB 图像
6. Resize 到 image_size × image_size
7. ToTensor，归一化到 [0, 1]
8. 返回 degraded_image, clean_image, file_name

作者：[Your Name]
日期：2026-01
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional


class Phase2Dataset(Dataset):
    """
    Phase 2 训练专用数据集
    
    特点：
    - 直接加载 input/label 配对
    - 不应用任何退化增强
    - 返回真实的混合退化数据
    """
    
    def __init__(self, 
                 input_dir: str,
                 label_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 return_path: bool = False):
        """
        Args:
            input_dir: 输入图像目录（真实退化数据）
            label_dir: 标签图像目录（GT）
            transform: 图像变换（resize, normalize等）
            return_path: 是否返回图像路径（用于保存结果）
        """
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.return_path = return_path
        
        # 获取所有图像文件
        self.input_images = sorted([
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        self.label_images = sorted([
            f for f in os.listdir(label_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        # 验证配对一致性
        assert len(self.input_images) == len(self.label_images), \
            f"❌ Input和Label数量不匹配: {len(self.input_images)} vs {len(self.label_images)}"
        
        # 验证文件名一致性（可选，但推荐）
        for inp, lab in zip(self.input_images, self.label_images):
            inp_name = os.path.splitext(inp)[0]
            lab_name = os.path.splitext(lab)[0]
            assert inp_name == lab_name, \
                f"❌ 文件名不匹配: {inp} vs {lab}"
        
        print(f"✅ Phase2Dataset 初始化成功")
        print(f"   - 输入目录: {input_dir}")
        print(f"   - 标签目录: {label_dir}")
        print(f"   - 图像数量: {len(self.input_images)}")
    
    def __len__(self) -> int:
        return len(self.input_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        返回：
        - degraded_image: (C, H, W) 真实退化图像
        - clean_image: (C, H, W) 干净图像
        - file_name: 文件名（不含扩展名）
        """
        # 构建完整路径
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        label_path = os.path.join(self.label_dir, self.label_images[idx])
        
        # 加载图像
        try:
            degraded_image = Image.open(input_path).convert("RGB")
            clean_image = Image.open(label_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"❌ 加载图像失败: {input_path}\n错误: {e}")
        
        # 应用变换
        if self.transform:
            degraded_image = self.transform(degraded_image)
            clean_image = self.transform(clean_image)
        
        # 获取文件名（不含扩展名）
        file_name = os.path.splitext(self.input_images[idx])[0]
        
        # 如果需要返回完整路径
        if self.return_path:
            return degraded_image, clean_image, input_path
        else:
            return degraded_image, clean_image, file_name


# ========== 便捷函数 ==========

def get_phase2_dataloader(
    input_dir: str,
    label_dir: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    return_path: bool = False
) -> DataLoader:
    """
    获取 Phase 2 数据加载器（便捷函数）
    
    Args:
        input_dir: 输入图像目录
        label_dir: 标签图像目录
        batch_size: 批大小
        image_size: 图像尺寸
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        pin_memory: 是否pin_memory（GPU加速）
        return_path: 是否返回完整路径
    
    Returns:
        DataLoader
    """
    # 定义变换（与 Phase 1 保持一致）
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # 自动归一化到 [0, 1]
    ])
    
    # 创建数据集
    dataset = Phase2Dataset(
        input_dir=input_dir,
        label_dir=label_dir,
        transform=transform,
        return_path=return_path
    )
    
    # 创建 DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(shuffle is True)  # 训练时丢弃不完整batch
    )


def get_dataloaders(
    train_input_dir: str,
    train_label_dir: str,
    val_input_dir: str,
    val_label_dir: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    同时获取训练和验证数据加载器
    
    Returns:
        (train_loader, val_loader)
    """
    print(f"\n{'='*60}")
    print("初始化 Phase 2 数据加载器")
    print(f"{'='*60}\n")
    
    # 训练集
    print("[1/2] 加载训练集...")
    train_loader = get_phase2_dataloader(
        input_dir=train_input_dir,
        label_dir=train_label_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        shuffle=True,  # 训练时打乱
        pin_memory=True
    )
    
    # 验证集
    print("\n[2/2] 加载验证集...")
    val_loader = get_phase2_dataloader(
        input_dir=val_input_dir,
        label_dir=val_label_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        shuffle=False,  # 验证时不打乱
        pin_memory=True
    )
    
    print(f"\n{'='*60}")
    print("✅ 数据加载器初始化完成")
    print(f"   - 训练批次: {len(train_loader)}")
    print(f"   - 验证批次: {len(val_loader)}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader


# ========== 数据验证工具 ==========

def validate_phase2_dataset(
    input_dir: str,
    label_dir: str,
    num_samples: int = 5
) -> None:
    """
    验证 Phase 2 数据集的正确性
    
    检查项：
    - 文件数量一致性
    - 文件名配对正确性
    - 图像尺寸
    - 数据范围
    """
    print(f"\n{'='*60}")
    print("Phase 2 数据集验证")
    print(f"{'='*60}\n")
    
    # 创建数据集
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = Phase2Dataset(
        input_dir=input_dir,
        label_dir=label_dir,
        transform=transform
    )
    
    print(f"[1/3] 基础信息:")
    print(f"   - 数据集大小: {len(dataset)}")
    
    # 采样检查
    print(f"\n[2/3] 采样检查 (前 {num_samples} 个样本):")
    
    for i in range(min(num_samples, len(dataset))):
        degraded, clean, file_name = dataset[i]
        
        print(f"\n   样本 {i+1}:")
        print(f"      文件名: {file_name}")
        print(f"      Degraded: shape={degraded.shape}, "
              f"range=[{degraded.min():.3f}, {degraded.max():.3f}]")
        print(f"      Clean: shape={clean.shape}, "
              f"range=[{clean.min():.3f}, {clean.max():.3f}]")
        
        # 验证数据范围
        assert degraded.min() >= 0.0 and degraded.max() <= 1.0, \
            f"❌ Degraded 数据超出 [0,1] 范围"
        assert clean.min() >= 0.0 and clean.max() <= 1.0, \
            f"❌ Clean 数据超出 [0,1] 范围"
    
    # DataLoader 测试
    print(f"\n[3/3] DataLoader 测试:")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    degraded_batch, clean_batch, file_names = next(iter(loader))
    
    print(f"   Batch shape: {degraded_batch.shape}")
    print(f"   File names: {file_names}")
    
    print(f"\n{'='*60}")
    print("✅ 数据集验证通过！")
    print(f"{'='*60}\n")


# ========== 测试代码 ==========

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Phase 2 Dataset 测试")
    print("="*60 + "\n")
    
    # ===== 配置测试路径 =====
    # 🔥 修改为你的实际路径
    TEST_INPUT_DIR = '/path/to/your/dataset/train/input'
    TEST_LABEL_DIR = '/path/to/your/dataset/train/label'
    
    # ===== 如果路径不存在，创建临时测试数据 =====
    if not os.path.exists(TEST_INPUT_DIR):
        print("⚠️  测试路径不存在，创建模拟数据...")
        
        import tempfile
        import shutil
        import numpy as np
        
        temp_dir = tempfile.mkdtemp()
        TEST_INPUT_DIR = os.path.join(temp_dir, 'input')
        TEST_LABEL_DIR = os.path.join(temp_dir, 'label')
        
        os.makedirs(TEST_INPUT_DIR)
        os.makedirs(TEST_LABEL_DIR)
        
        # 创建 10 张模拟图像
        for i in range(10):
            # Clean image (label)
            clean = np.random.randint(150, 255, (256, 256, 3), dtype=np.uint8)
            clean_img = Image.fromarray(clean)
            clean_img.save(os.path.join(TEST_LABEL_DIR, f'img_{i:03d}.png'))
            
            # Degraded image (input) - 添加一些退化效果
            degraded = (clean * 0.7 + 50).clip(0, 255).astype(np.uint8)
            degraded_img = Image.fromarray(degraded)
            degraded_img.save(os.path.join(TEST_INPUT_DIR, f'img_{i:03d}.png'))
        
        print(f"✅ 创建临时测试数据于: {temp_dir}\n")
    
    # ===== 测试 1: 数据集验证 =====
    print("[测试 1] 数据集验证")
    print("-" * 60)
    
    try:
        validate_phase2_dataset(
            input_dir=TEST_INPUT_DIR,
            label_dir=TEST_LABEL_DIR,
            num_samples=3
        )
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== 测试 2: DataLoader 功能 =====
    print("\n[测试 2] DataLoader 功能")
    print("-" * 60)
    
    try:
        loader = get_phase2_dataloader(
            input_dir=TEST_INPUT_DIR,
            label_dir=TEST_LABEL_DIR,
            batch_size=4,
            image_size=256,
            num_workers=0,  # 测试时用单线程
            shuffle=True
        )
        
        # 测试迭代
        for batch_idx, (degraded, clean, file_names) in enumerate(loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Degraded: {degraded.shape}, "
                  f"range=[{degraded.min():.3f}, {degraded.max():.3f}]")
            print(f"  Clean: {clean.shape}, "
                  f"range=[{clean.min():.3f}, {clean.max():.3f}]")
            print(f"  Files: {file_names}")
            
            if batch_idx >= 1:  # 只测试前 2 个 batch
                break
        
        print("\n✅ DataLoader 测试通过")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== 测试 3: get_dataloaders() 函数 =====
    print("\n[测试 3] get_dataloaders() 函数")
    print("-" * 60)
    
    try:
        # 创建临时验证集目录
        val_input_dir = TEST_INPUT_DIR  # 复用训练集
        val_label_dir = TEST_LABEL_DIR
        
        train_loader, val_loader = get_dataloaders(
            train_input_dir=TEST_INPUT_DIR,
            train_label_dir=TEST_LABEL_DIR,
            val_input_dir=val_input_dir,
            val_label_dir=val_label_dir,
            batch_size=4,
            image_size=256,
            num_workers=0
        )
        
        print(f"\n训练集:")
        degraded, clean, _ = next(iter(train_loader))
        print(f"  Shape: {degraded.shape}")
        
        print(f"\n验证集:")
        degraded, clean, _ = next(iter(val_loader))
        print(f"  Shape: {degraded.shape}")
        
        print("\n✅ get_dataloaders() 测试通过")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== 清理临时文件 =====
    if 'temp_dir' in locals():
        shutil.rmtree(temp_dir)
        print(f"\n✅ 清理临时文件")
    
    print("\n" + "="*60)
    print("✅ 所有测试完成！")
    print("="*60 + "\n")
