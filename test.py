import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from DSC_ICME_NAS import DSC_Net


# ============================================================
# 1. 直接在这里修改路径和参数
# ============================================================

# U45 无配对数据集 input 文件夹
U45_INPUT_DIR = "dataset/U45_input"

# U45 增强结果保存路径
U45_OUTPUT_DIR = "test/U45-145"

# UIEB 数据集训练好的预训练权重
CHECKPOINT_PATH = "results5_channel=32/UIEB/checkpoints/epoch_145_22.59.pth"

# 测试输入尺寸，需要和训练时保持一致
IMG_SIZE = 256

# 是否把输出结果恢复成原图尺寸
RESTORE_ORIGINAL_SIZE = True

# 使用设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. 工具函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_images(input_dir):
    """
    收集 U45 文件夹下的所有图像。
    """
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    image_paths = []

    input_dir = Path(input_dir)

    for ext in exts:
        image_paths.extend(input_dir.glob(f"*{ext}"))
        image_paths.extend(input_dir.glob(f"*{ext.upper()}"))

    image_paths = sorted(list(set(image_paths)))
    return image_paths


def tensor_to_pil(tensor):
    """
    tensor: [1, 3, H, W] 或 [3, H, W]
    范围: [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    return transforms.ToPILImage()(tensor)


def load_checkpoint(model, checkpoint_path, device):
    """
    加载 UIEB 预训练权重。
    兼容：
    1. checkpoint 中包含 model_state_dict
    2. checkpoint 本身就是 state_dict
    3. DataParallel 保存的 module. 前缀
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Loaded checkpoint['model_state_dict']")
    else:
        state_dict = checkpoint
        print("Loaded raw state_dict")

    # 去掉 DataParallel 的 module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)

    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", None)
        psnr = checkpoint.get("val_psnr", None)
        ssim = checkpoint.get("val_ssim", None)

        if epoch is not None:
            print(f"Checkpoint epoch: {epoch + 1}")
        if psnr is not None:
            print(f"Checkpoint PSNR: {psnr:.4f} dB")
        if ssim is not None:
            print(f"Checkpoint SSIM: {ssim:.6f}")

    print("Checkpoint loaded successfully.\n")
    return model


# ============================================================
# 3. U45 测试主函数
# ============================================================

@torch.no_grad()
def test_u45():
    print("=" * 60)
    print("Testing UIEB pretrained model on U45 dataset")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"U45 input dir: {U45_INPUT_DIR}")
    print(f"Output dir: {U45_OUTPUT_DIR}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Restore original size: {RESTORE_ORIGINAL_SIZE}")
    print("=" * 60)

    ensure_dir(U45_OUTPUT_DIR)

    # 1. 初始化模型
    model = DSC_Net().to(DEVICE)

    # 2. 加载 UIEB 预训练权重
    model = load_checkpoint(model, CHECKPOINT_PATH, DEVICE)
    model.eval()

    # 3. 收集 U45 输入图像
    image_paths = collect_images(U45_INPUT_DIR)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {U45_INPUT_DIR}")

    print(f"Found {len(image_paths)} images.\n")

    # 4. 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # 5. 逐张增强并保存
    for img_path in tqdm(image_paths, desc="Enhancing U45"):
        img = Image.open(img_path).convert("RGB")
        original_size = img.size  # (W, H)

        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        input_tensor = torch.nan_to_num(
            input_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        ).clamp(0.0, 1.0)

        # 模型推理
        pred = model(input_tensor)
        pred = torch.nan_to_num(
            pred,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        ).clamp(0.0, 1.0)

        # 转 PIL
        pred_img = tensor_to_pil(pred)

        # 是否恢复到原图尺寸
        if RESTORE_ORIGINAL_SIZE:
            pred_img = pred_img.resize(original_size, Image.BICUBIC)

        # 保存，文件名保持和输入一致
        save_path = os.path.join(U45_OUTPUT_DIR, img_path.name)
        pred_img.save(save_path)

    print("\n" + "=" * 60)
    print("U45 enhancement finished.")
    print(f"Enhanced images saved to: {U45_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    test_u45()