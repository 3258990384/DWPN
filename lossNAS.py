"""
lossNAS.py
==========

统一损失函数库(支持 Phase 1 和 Phase 2)

Phase 1 损失:
- 专家先验引导损失(Dark Channel / Gray World / Illumination)
- 用于专家分化训练

Phase 2 损失:
- 统一质量损失(Charbonnier + SSIM + Gradient + Color)
- 用于动态级联协同训练

修复版本:
✅ GradientLoss 的 Sobel 核注册为 buffer
✅ SSIMLoss 使用 utils.calculate_ssim
✅ ColorLoss 使用 utils.rgb2lab(无需 kornia)

作者: [Your Name]
日期: 2026-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calculate_ssim, rgb2lab    # ✅ 确保导入了你的工具函数
import numpy as np


# ========== 基础损失组件 ==========

class CharbonnierLoss(nn.Module):
    """
    Charbonnier 损失(平滑 L1)
    
    L = mean(sqrt((pred - target)^2 + epsilon^2))
    
    优势:
    - 比 L1 更平滑(可导)
    - 比 MSE 对异常值更鲁棒
    - 训练后期更稳定
    """
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()


class SSIMLoss(nn.Module):
    """
    使用 utils.py 中的 torchmetrics 实现 SSIM Loss
    
    ✅ 修复: 调用 utils.calculate_ssim
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ✅ 直接调用你的 utils 函数
        # calculate_ssim 返回的是均值 SSIM (0~1)
        ssim_val = calculate_ssim(pred, target)
        
        # 返回 1 - SSIM 作为损失
        return 1.0 - ssim_val


class GradientLoss(nn.Module):
    """
    梯度损失(边缘一致性)
    
    ✅ 修复: Sobel 核注册为 buffer,自动跟随模型移动到 GPU
    """
    def __init__(self):
        super().__init__()
        
        # ✅ 定义 Sobel 卷积核
        sobel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).reshape(3, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).reshape(3, 1, 3, 3)
        
        # ✅ 注册为 buffer(会自动跟随模型移动设备)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W)
            target: (B, 3, H, W)
        
        Returns:
            loss: 标量
        """
        # ✅ 现在 self.sobel_x 和 self.sobel_y 会自动在正确的设备上
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


class ColorLoss(nn.Module):
    """
    Lab 色度损失
    
    ✅ 修复: 使用 utils.rgb2lab(无需 kornia)
    
    在 Lab 色空间的 a/b 通道上计算损失
    
    优势:
    - 色度与亮度分离
    - 符合人类视觉感知
    - 避免 RGB 均值匹配的亮度干扰
    """
    def __init__(self, w_L=0.0, w_ab=1.0):
        """
        Args:
            w_L: 亮度权重(默认 0,只关注色度)
            w_ab: 色度权重
        """
        super().__init__()
        self.w_L = w_L
        self.w_ab = w_ab
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 3, H, W) 预测图像,范围 [0, 1]
            target: (B, 3, H, W) 目标图像,范围 [0, 1]
        
        Returns:
            loss: 标量,颜色损失
        """
        # ✅ RGB → Lab (使用你的 utils 函数)
        # rgb2lab 返回归一化后的 Lab: (B, 3, H, W)
        # 通道 0: L (亮度), 通道 1-2: a, b (色度)
        pred_lab = rgb2lab(pred)
        target_lab = rgb2lab(target)
        
        # 分离 L 和 ab 通道
        pred_L = pred_lab[:, 0:1, :, :]    # (B, 1, H, W)
        pred_ab = pred_lab[:, 1:3, :, :]   # (B, 2, H, W)
        
        target_L = target_lab[:, 0:1, :, :]
        target_ab = target_lab[:, 1:3, :, :]
        
        # 计算损失
        loss_L = F.l1_loss(pred_L, target_L) if self.w_L > 0 else 0.0
        loss_ab = F.l1_loss(pred_ab, target_ab)
        
        return self.w_L * loss_L + self.w_ab * loss_ab


class ColorLossYCbCr(nn.Module):
    """
    YCbCr 色度损失(无需 kornia 的备选方案)
    
    在 YCbCr 色空间的 Cb/Cr 通道上计算损失
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # RGB → YCbCr(近似公式)
        # Pred
        pred_y = 0.299*pred[:,0] + 0.587*pred[:,1] + 0.114*pred[:,2]
        pred_cb = -0.169*pred[:,0] - 0.331*pred[:,1] + 0.5*pred[:,2]
        pred_cr = 0.5*pred[:,0] - 0.419*pred[:,1] - 0.081*pred[:,2]
        
        # Target
        target_y = 0.299*target[:,0] + 0.587*target[:,1] + 0.114*target[:,2]
        target_cb = -0.169*target[:,0] - 0.331*target[:,1] + 0.5*target[:,2]
        target_cr = 0.5*target[:,0] - 0.419*target[:,1] - 0.081*target[:,2]
        
        # 只对色度通道 Cb, Cr 计算损失
        loss = (pred_cb - target_cb).abs().mean() + \
               (pred_cr - target_cr).abs().mean()
        
        return loss


# ========== Phase 1 专家先验损失 ==========

class DarkChannelLoss(nn.Module):
    """
    暗通道损失(去散射专家先验)
    
    引导 Scatter Expert 学习去雾能力
    
    原理:
    - 无雾图像的暗通道接近 0
    - 有雾图像的暗通道较大
    
    目标:最小化输出的暗通道值
    """
    def __init__(self, patch_size=15):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, pred):
        """
        Args:
            pred: (B, 3, H, W) 输出图像
        
        Returns:
            暗通道损失(越小越好)
        """
        # 1. 取 RGB 三通道最小值
        dark_channel, _ = pred.min(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # 2. 最小值池化(模拟局部 patch)
        kernel_size = self.patch_size
        dark_channel = F.max_pool2d(
            -dark_channel,  # 负号: max_pool(-x) = -min_pool(x)
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        dark_channel = -dark_channel
        
        # 3. 返回均值(鼓励接近 0)
        return dark_channel.mean()


class GrayWorldLoss(nn.Module):
    """
    灰度世界损失(色偏校正专家先验)
    
    引导 Color Expert 学习白平衡能力
    
    原理:
    - 自然图像的 RGB 均值应该接近
    - 色偏图像的 RGB 均值有明显偏差
    
    目标:最小化 RGB 通道均值的方差
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred):
        """
        Args:
            pred: (B, 3, H, W) 输出图像
        
        Returns:
            灰度世界损失(越小越好)
        """
        # 计算每个通道的均值
        r_mean = pred[:, 0].mean()
        g_mean = pred[:, 1].mean()
        b_mean = pred[:, 2].mean()
        
        # 全局均值
        global_mean = (r_mean + g_mean + b_mean) / 3.0
        
        # 计算方差(鼓励 RGB 均值接近)
        loss = (r_mean - global_mean).abs() + \
               (g_mean - global_mean).abs() + \
               (b_mean - global_mean).abs()
        
        return loss


class IlluminationSmoothLoss(nn.Module):
    """
    光照平滑损失(暗光增强专家先验)
    
    引导 LowLight Expert 学习平滑增强能力
    
    原理:
    - 光照分量应该平滑变化
    - 避免过度增强导致的噪声放大
    
    目标:最小化亮度通道的梯度
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred):
        """
        Args:
            pred: (B, 3, H, W) 输出图像
        
        Returns:
            光照平滑损失(越小越好)
        """
        # 1. 计算亮度通道(Y = 0.299*R + 0.587*G + 0.114*B)
        luminance = 0.299*pred[:,0] + 0.587*pred[:,1] + 0.114*pred[:,2]
        luminance = luminance.unsqueeze(1)  # (B, 1, H, W)
        
        # 2. 计算梯度(TV 正则)
        # 水平梯度
        grad_x = torch.abs(luminance[:, :, :, :-1] - luminance[:, :, :, 1:])
        # 垂直梯度
        grad_y = torch.abs(luminance[:, :, :-1, :] - luminance[:, :, 1:, :])
        
        # 3. 返回总变差(鼓励平滑)
        return grad_x.mean() + grad_y.mean()


# ========== Phase 1 统一损失 ==========

class UnifiedLossPhase1(nn.Module):
    """
    Phase 1 统一损失(专家分化训练)
    
    组成:
    - L1 损失(主监督)
    - SSIM 损失(结构一致性)
    - 专家先验损失(可选,用于引导分化)
    
    用法:
    - Scatter Expert: l1 + ssim + dark_channel
    - Color Expert: l1 + ssim + gray_world
    - LowLight Expert: l1 + ssim + illum_smooth
    """
    def __init__(
        self,
        l1_weight=1.0,
        ssim_weight=0.2,
        dark_channel_weight=0.0,
        gray_world_weight=0.0,
        illum_smooth_weight=0.0
    ):
        super().__init__()
        
        self.l1_w = l1_weight
        self.ssim_w = ssim_weight
        self.dc_w = dark_channel_weight
        self.gw_w = gray_world_weight
        self.is_w = illum_smooth_weight
        
        # 基础损失
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        
        # 专家先验损失
        if dark_channel_weight > 0:
            self.dark_channel = DarkChannelLoss()
        
        if gray_world_weight > 0:
            self.gray_world = GrayWorldLoss()
        
        if illum_smooth_weight > 0:
            self.illum_smooth = IlluminationSmoothLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 3, H, W) 预测图像
            target: (B, 3, H, W) 目标图像
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 主监督损失
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target)
        
        total_loss = self.l1_w * loss_l1 + self.ssim_w * loss_ssim
        
        loss_dict = {
            'l1': loss_l1.item(),
            'ssim': loss_ssim.item(),
        }
        
        # 专家先验损失
        if self.dc_w > 0:
            loss_dc = self.dark_channel(pred)
            total_loss += self.dc_w * loss_dc
            loss_dict['dark_channel'] = loss_dc.item()
        
        if self.gw_w > 0:
            loss_gw = self.gray_world(pred)
            total_loss += self.gw_w * loss_gw
            loss_dict['gray_world'] = loss_gw.item()
        
        if self.is_w > 0:
            loss_is = self.illum_smooth(pred)
            total_loss += self.is_w * loss_is
            loss_dict['illum_smooth'] = loss_is.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ========== Phase 2 统一损失 ==========

class UnifiedLossPhase2(nn.Module):
    """
    Phase 2 统一损失(动态级联协同训练)
    
    组成:
    - Charbonnier 损失(主像素损失)
    - SSIM 损失(结构一致性)
    - Gradient 损失(边缘/纹理)
    - Color 损失(色彩一致性)
    - Aux 损失(可选中间深监督)
    
    配置:
    - 配置 A(PSNR 优先): char=1.0, ssim=0.2, grad=0.05, color=0.02
    - 配置 B(平衡质量): char=1.0, ssim=0.3, grad=0.06, color=0.03
    """
    def __init__(
        self,
        charbonnier_weight=1.0,
        ssim_weight=0.2,
        gradient_weight=0.05,
        color_weight=0.02,
        aux_weights=[0.05, 0.1],  # 第 1 步、第 2 步的深监督权重
        use_aux=True,
        use_ycbcr_color=False  # 是否使用 YCbCr(不需要 kornia)
    ):
        super().__init__()
        
        self.char_w = charbonnier_weight
        self.ssim_w = ssim_weight
        self.grad_w = gradient_weight
        self.color_w = color_weight
        self.aux_w = aux_weights
        self.use_aux = use_aux
        
        # 损失函数
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.gradient = GradientLoss()
        
        # ✅ 色彩损失(根据是否需要 kornia 选择)
        if use_ycbcr_color:
            self.color = ColorLossYCbCr()
        else:
            self.color = ColorLoss()  # 使用 utils.rgb2lab
    
    def forward(self, pred, target, aux_images=None):
        """
        Args:
            pred: (B, 3, H, W) 最终输出
            target: (B, 3, H, W) GT
            aux_images: List[(B,3,H,W)] 中间结果(可选)
                - images[0] = 输入
                - images[1] = 第 1 步输出
                - images[2] = 第 2 步输出
                - images[3] = 第 3 步输出 = pred
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典(用于日志)
        """
        # ===== 主损失(最终输出) =====
        loss_char = self.charbonnier(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_grad = self.gradient(pred, target)
        loss_color = self.color(pred, target)
        
        total_loss = self.char_w * loss_char + \
                     self.ssim_w * loss_ssim + \
                     self.grad_w * loss_grad + \
                     self.color_w * loss_color
        
        loss_dict = {
            'char': loss_char.item(),
            'ssim': loss_ssim.item(),
            'grad': loss_grad.item(),
            'color': loss_color.item(),
        }
        
        # ===== 中间深监督(可选) =====
        if self.use_aux and aux_images is not None and len(aux_images) >= 3:
            aux_loss = 0.0
            
            # 只对前 2 步(images[1] 和 images[2])
            # images[3] = 最终输出,已经包含在主损失里
            for i, weight in enumerate(self.aux_w):
                if i + 1 < len(aux_images) - 1:  # 不包括最后一个(最终输出)
                    aux_img = aux_images[i + 1]
                    aux_loss += weight * self.charbonnier(aux_img, target)
            
            total_loss += aux_loss
            loss_dict['aux'] = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        else:
            loss_dict['aux'] = 0.0
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict



# ========== 新论文：空频一致性损失 ==========

# ========== 论文一：FTD-WaveUIE 损失函数 ==========

class GrayWorldBalanceLoss(nn.Module):
    """
    灰世界颜色平衡损失。

    作用:
        约束输出图像 RGB 三通道均值接近，
        缓解水下蓝绿色偏色。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        mean_rgb = x.mean(dim=(2, 3))              # [B, 3]
        mean_gray = mean_rgb.mean(dim=1, keepdim=True)

        loss = torch.mean(torch.abs(mean_rgb - mean_gray))
        return loss


class IlluminationSmoothLossFTD(nn.Module):
    """
    光照平滑损失。

    作用:
        避免增强后出现局部亮度突变，
        让低频亮度恢复更稳定。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # RGB -> luminance
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        loss_h = torch.mean(torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :]))
        loss_w = torch.mean(torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1]))

        return loss_h + loss_w


class FrequencyConsistencyLoss(nn.Module):
    """
    频率一致性损失。

    L_freq = || log(1 + |FFT(pred)|) - log(1 + |FFT(gt)|) ||_1

    作用:
        约束输出和 GT 的频谱幅值分布一致，
        避免过平滑或高频噪声放大。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1), norm='ortho')

        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))

        loss = F.l1_loss(pred_mag, target_mag)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=0.0)

        return loss


class FTDWaveUIELoss(nn.Module):
    """
    FTD-WaveUIE 总损失。

    L = L_rec + lambda_f L_freq + lambda_c L_color + lambda_i L_illum

    默认权重比较保守，避免训练初期 PSNR 大幅波动或 NaN。
    """
    def __init__(self,
                 rec_weight=1.0,
                 freq_weight=0.01,
                 color_weight=0.002,
                 illum_weight=0.002,
                 use_charbonnier=True):
        super().__init__()

        self.rec_w = rec_weight
        self.freq_w = freq_weight
        self.color_w = color_weight
        self.illum_w = illum_weight
        self.use_charbonnier = use_charbonnier

        self.charbonnier = CharbonnierLoss(epsilon=1e-3)
        self.freq_loss = FrequencyConsistencyLoss()
        self.gray_world = GrayWorldBalanceLoss()
        self.illum_smooth = IlluminationSmoothLossFTD()

    def forward(self, pred, target):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        if self.use_charbonnier:
            loss_rec = self.charbonnier(pred, target)
        else:
            loss_rec = F.l1_loss(pred, target)

        loss_freq = self.freq_loss(pred, target)
        loss_color = self.gray_world(pred)
        loss_illum = self.illum_smooth(pred)

        total = (
            self.rec_w * loss_rec +
            self.freq_w * loss_freq +
            self.color_w * loss_color +
            self.illum_w * loss_illum
        )

        total = torch.nan_to_num(total, nan=0.0, posinf=1e3, neginf=0.0)

        loss_dict = {
            'rec': float(loss_rec.detach().item()),
            'freq': float(loss_freq.detach().item()),
            'color_balance': float(loss_color.detach().item()),
            'illum': float(loss_illum.detach().item()),
            'total': float(total.detach().item()),

            # 为了兼容 train_phase1_wave.py 里原来的 loss_dict.get('band', 0.0)
            'band': 0.0,
            'ssim': 0.0
        }

        return total, loss_dict


def get_loss_wave_phase1():
    """
    论文一 FTD-WaveUIE 单阶段训练损失。

    注意:
        freq_weight 不建议一开始太大。
        如果训练稳定，后面可以从 0.02 提到 0.03 或 0.05。
    """
    return FTDWaveUIELoss(
        rec_weight=1.0,
        freq_weight=0.01,
        color_weight=0.002,
        illum_weight=0.002,
        use_charbonnier=True
    )




# ========== 工具函数 ==========

def get_loss_phase1(expert_type='scatter'):
    """
    获取 Phase 1 损失函数(根据专家类型)
    
    Args:
        expert_type: 'scatter' / 'color' / 'lowlight'
    
    Returns:
        UnifiedLossPhase1 实例
    """
    if expert_type == 'scatter':
        return UnifiedLossPhase1(
            l1_weight=1.0,
            ssim_weight=0.2,
            dark_channel_weight=0.1  # 去散射先验
        )
    
    elif expert_type == 'color':
        return UnifiedLossPhase1(
            l1_weight=1.0,
            ssim_weight=0.2,
            gray_world_weight=0.05  # 色偏校正先验
        )
    
    elif expert_type == 'lowlight':
        return UnifiedLossPhase1(
            l1_weight=1.0,
            ssim_weight=0.2,
            illum_smooth_weight=0.03  # 暗光增强先验
        )
    
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")


def get_loss_phase2(config='A'):
    """
    获取 Phase 2 损失函数(根据配置)
    
    Args:
        config: 'A' (PSNR 优先) / 'B' (平衡质量)
    
    Returns:
        UnifiedLossPhase2 实例
    """
    if config == 'A':
        # 配置 A: PSNR 优先
        return UnifiedLossPhase2(
            charbonnier_weight=1.0,
            ssim_weight=0.2,
            gradient_weight=0.05,
            color_weight=0.02,
            aux_weights=[0.05, 0.1],
            use_aux=True
        )
    
    elif config == 'B':
        # 配置 B: 平衡 PSNR 和主观质量
        return UnifiedLossPhase2(
            charbonnier_weight=1.0,
            ssim_weight=0.3,
            gradient_weight=0.06,
            color_weight=0.03,
            aux_weights=[0.05, 0.1],
            use_aux=True
        )
    
    else:
        raise ValueError(f"Unknown config: {config}")


# ========== 测试代码 ==========

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Loss Function Tests")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 测试数据
    pred = torch.rand(2, 3, 256, 256, device=device)
    target = torch.rand(2, 3, 256, 256, device=device)
    aux_images = [
        torch.rand(2, 3, 256, 256, device=device),  # 输入
        torch.rand(2, 3, 256, 256, device=device),  # 第 1 步
        torch.rand(2, 3, 256, 256, device=device),  # 第 2 步
        pred  # 第 3 步 = 最终输出
    ]
    
    # ===== 测试 Phase 1 损失 =====
    print("[Test 1] Phase 1 Losses:")
    
    for expert_type in ['scatter', 'color', 'lowlight']:
        loss_fn = get_loss_phase1(expert_type).to(device)
        loss, loss_dict = loss_fn(pred, target)
        
        print(f"\n  {expert_type.capitalize()} Expert:")
        print(f"    Total: {loss_dict['total']:.4f}")
        print(f"    L1: {loss_dict['l1']:.4f}")
        print(f"    SSIM: {loss_dict['ssim']:.4f}")
        
        if 'dark_channel' in loss_dict:
            print(f"    Dark Channel: {loss_dict['dark_channel']:.4f}")
        if 'gray_world' in loss_dict:
            print(f"    Gray World: {loss_dict['gray_world']:.4f}")
        if 'illum_smooth' in loss_dict:
            print(f"    Illum Smooth: {loss_dict['illum_smooth']:.4f}")
    
    # ===== 测试 Phase 2 损失 =====
    print("\n[Test 2] Phase 2 Losses:")
    
    for config in ['A', 'B']:
        loss_fn = get_loss_phase2(config).to(device)
        loss, loss_dict = loss_fn(pred, target, aux_images)
        
        print(f"\n  Config {config}:")
        print(f"    Total: {loss_dict['total']:.4f}")
        print(f"    Charbonnier: {loss_dict['char']:.4f}")
        print(f"    SSIM: {loss_dict['ssim']:.4f}")
        print(f"    Gradient: {loss_dict['grad']:.4f}")
        print(f"    Color: {loss_dict['color']:.4f}")
        print(f"    Aux: {loss_dict['aux']:.4f}")
    
    # ===== 测试梯度传播 =====
    print("\n[Test 3] Gradient Flow:")
    
    pred.requires_grad = True
    loss_fn = get_loss_phase2('A').to(device)
    loss, _ = loss_fn(pred, target, aux_images)
    loss.backward()
    
    assert pred.grad is not None, "❌ Gradient flow failed!"
    print(f"  ✅ Gradient mean: {pred.grad.mean().item():.6f}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")

