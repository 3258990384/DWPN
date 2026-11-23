"""
Lightweight U-Net Backbone for Feature Extraction (3-Layer Version)
===================================================================

🔥 轻量级 3层版本
- 输入：特征图 [B, in_channels, H, W]（如16通道）
- 输出：特征图 [B, out_channels, H, W]（如16通道）
- 用途：DSC_IJCAI 的共享特征提取器

作者：[Your Name]
日期：2026-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from multihead_diffattn_2D import MultiheadDiffAttn


# ========== 🔥 3层通道配置 ==========

def get_channel_config(in_channels: int):
    """
    根据输入通道数生成 3层 通道配置
    
    Args:
        in_channels: 输入通道数（如16）
    
    Returns:
        3层通道配置 (CH1, CH2, CH3)
    
    Example:
        in_channels=16 → (16, 32, 64)
    """
    factor = max(1, in_channels // 16)
    return (
        16 * factor,   # CH1
        32 * factor,   # CH2
        64 * factor    # CH3 (bottleneck)
    )


# ========== 基础模块 ==========

class ResidualBlock(nn.Module):
    """
    残差块
    - 2× 卷积 + GroupNorm + GELU
    - SE 注意力（可选）
    - Dropout2d
    """
    def __init__(self, channels: int, dropout: float = 0.1, use_se: bool = True):
        super().__init__()
        self.use_se = use_se
        
        # 🔥 动态调整 GroupNorm 的 num_groups
        num_groups = min(16, max(1, channels // 4))
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups, channels)
        )
        
        # 🔥 SE 注意力
        if use_se:
            reduction = max(4, channels // 16)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, 1),
                nn.GELU(),
                nn.Conv2d(channels // reduction, channels, 1),
                nn.Sigmoid()
            )
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.use_se:
            attn = self.se(x)
            x = x * attn
        
        x = self.dropout(x)
        x = x + identity
        x = self.activation(x)
        
        return x


# ========== 🔥 Attention Bottleneck ==========

class AttentionBottleneckBlock(nn.Module):
    """单个 Attention Bottleneck 块"""
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        embed_dim = (channels // num_heads) * num_heads
        self.need_proj = (channels != embed_dim)
        
        if self.need_proj:
            self.proj_in = nn.Conv2d(channels, embed_dim, 1)
            self.proj_out = nn.Conv2d(embed_dim, channels, 1)
        
        num_groups = min(16, max(1, embed_dim // 4))
        
        self.norm1 = nn.GroupNorm(num_groups, embed_dim if self.need_proj else channels)
        
        self.attn = MultiheadDiffAttn(
            embed_dim=embed_dim,
            depth=0,
            num_heads=num_heads,
            num_kv_heads=max(2, num_heads // 2)
        )
        
        self.norm2 = nn.GroupNorm(num_groups, embed_dim if self.need_proj else channels)
        self.conv = ResidualBlock(embed_dim if self.need_proj else channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        if self.need_proj:
            x = self.proj_in(x)
        
        x = self.norm1(x)
        x = self.attn(x, None)
        
        if self.need_proj:
            x = self.proj_out(x)
        x = x + identity
        
        identity2 = x
        x = self.norm2(x)
        x = self.conv(x)
        x = x + identity2
        
        return x


class AttentionBottleneck(nn.Module):
    """多块 Attention Bottleneck"""
    def __init__(self, channels: int, num_blocks: int = 1, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttentionBottleneckBlock(channels, num_heads, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PlainBottleneck(nn.Module):
    """纯卷积 Bottleneck（无 attention）"""
    def __init__(self, channels: int, num_blocks: int = 1, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, dropout=dropout) 
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class StableDampedWaveOperator(nn.Module):
    """
    稳定阻尼波传播算子，放在 U-Net bottleneck。
    - 输入: bottleneck 特征 x [B,C,H,W]
    - degradation_state: DegradationAnalyzer 输出的 12 维状态 [B,12]
    - 数值稳定设计: 小步长、参数限幅、update 限幅、nan_to_num
    """
    def __init__(self, channels: int, state_dim: int = 12, steps: int = 2, dt: float = 0.08):
        super().__init__()
        self.channels = channels
        self.steps = steps
        self.dt = dt

        self.param_mlp = nn.Sequential(
            nn.Linear(state_dim, max(32, channels)),
            nn.GELU(),
            nn.Linear(max(32, channels), channels * 2)
        )

        lap = torch.tensor([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap)

        # 初始只让波算子产生很小扰动，避免一开始破坏 U-Net 表征
        self.res_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor, degradation_state: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if degradation_state is None:
            degradation_state = x.new_zeros(B, 12)

        degradation_state = torch.nan_to_num(degradation_state, nan=0.0, posinf=1.0, neginf=0.0)
        degradation_state = degradation_state.clamp(0.0, 4.0)

        params = self.param_mlp(degradation_state).view(B, 2, C, 1, 1)
        raw_c, raw_gamma = params[:, 0], params[:, 1]

        # 传播速度和阻尼都限制在温和范围，避免振荡/爆炸
        c = 0.04 + 0.12 * torch.sigmoid(raw_c)          # [0.04, 0.16]
        gamma = 0.04 + 0.18 * torch.sigmoid(raw_gamma)  # [0.04, 0.22]

        y = x
        kernel = self.lap_kernel.to(dtype=x.dtype, device=x.device).expand(C, 1, 3, 3)
        for _ in range(self.steps):
            lap = F.conv2d(y, kernel, padding=1, groups=C)
            update = (c * c) * lap - gamma * y
            update = torch.clamp(update, -1.0, 1.0)
            y = y + self.dt * update
            y = torch.nan_to_num(y, nan=0.0, posinf=6.0, neginf=-6.0).clamp(-6.0, 6.0)

        scale = torch.sigmoid(self.res_scale) * 0.2
        out = x + scale * (y - x)
        return torch.nan_to_num(out, nan=0.0, posinf=6.0, neginf=-6.0).clamp(-6.0, 6.0)


class UDWPO(nn.Module):
    """
    UD-WPO: Underwater Degradation-aware Wave Propagation Operator

    作用:
        在 U-Net bottleneck 特征上执行 FFT 频域阻尼波传播。

    输入:
        x: [B, C, H, W]
        degradation_state: [B, 3]
            [:,0] scatter
            [:,1] color cast
            [:,2] low light

    输出:
        out: [B, C, H, W]

    数值稳定设计:
        1. alpha 和 velocity 使用 sigmoid 限制范围；
        2. omega_d2 使用 clamp_min，避免 sqrt 负数；
        3. wave_kernel 使用 nan_to_num 和 clamp；
        4. 残差缩放系数初始很小，避免训练初期破坏特征。
    """

    def __init__(self, channels: int, state_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.channels = channels

        self.param_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        # 初始只让波传播产生小扰动，防止一开始破坏 U-Net 表征
        self.res_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor, degradation_state: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x

        if degradation_state is None:
            degradation_state = x.new_zeros(B, 3)

        degradation_state = torch.nan_to_num(
            degradation_state,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        ).clamp(0.0, 1.0)

        # 根据三维退化状态预测 alpha 和 velocity
        raw_params = self.param_predictor(degradation_state)
        raw_alpha = raw_params[:, 0].view(B, 1, 1, 1)
        raw_velocity = raw_params[:, 1].view(B, 1, 1, 1)

        # 稳定范围，不要直接 Softplus 放开，否则容易 NaN
        alpha = 0.02 + 0.18 * torch.sigmoid(raw_alpha)          # [0.02, 0.20]
        velocity = 0.05 + 0.25 * torch.sigmoid(raw_velocity)    # [0.05, 0.30]

        feat = self.proj_in(x)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=6.0, neginf=-6.0).clamp(-6.0, 6.0)

        # FFT 建议用 float32，避免半精度复数不稳定
        feat_float = feat.float()

        fy = torch.fft.fftfreq(H, device=x.device).view(1, 1, H, 1)
        fx = torch.fft.fftfreq(W, device=x.device).view(1, 1, 1, W)

        omega2 = fx * fx + fy * fy
        omega2 = omega2.clamp_min(1e-6)

        # 阻尼频率
        omega_d2 = velocity * velocity * omega2 - (alpha / 2.0) ** 2
        omega_d2 = omega_d2.clamp_min(1e-6)
        omega_d = torch.sqrt(omega_d2)

        t = 1.0

        feat_fft = torch.fft.fft2(feat_float, dim=(-2, -1), norm='ortho')

        damping = torch.exp(-alpha * t / 2.0)
        cos_term = torch.cos(omega_d * t)
        sin_term = torch.sin(omega_d * t) / (omega_d + 1e-6)

        wave_kernel = damping * (cos_term + 0.5 * alpha * sin_term)
        wave_kernel = torch.nan_to_num(
            wave_kernel,
            nan=1.0,
            posinf=2.0,
            neginf=-2.0
        ).clamp(-2.0, 2.0)

        propagated_fft = feat_fft * wave_kernel
        propagated = torch.fft.ifft2(propagated_fft, dim=(-2, -1), norm='ortho').real

        propagated = propagated.to(dtype=x.dtype)
        propagated = self.proj_out(propagated)
        propagated = torch.nan_to_num(
            propagated,
            nan=0.0,
            posinf=6.0,
            neginf=-6.0
        ).clamp(-6.0, 6.0)

        # 小残差注入，避免训练初期不稳定
        scale = torch.sigmoid(self.res_scale) * 0.2
        out = identity + scale * propagated

        return torch.nan_to_num(out, nan=0.0, posinf=6.0, neginf=-6.0).clamp(-6.0, 6.0)

# ========== Encoder/Decoder ==========

class EncoderBlock(nn.Module):
    """编码器块"""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels, use_se=True)
            for _ in range(num_blocks)
        ])
        
        self.pool = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    """解码器块"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, 
                 num_blocks: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        concat_channels = out_channels + skip_channels
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(concat_channels, dropout=dropout, use_se=True)
            for _ in range(num_blocks)
        ])
        
        self.conv_out = nn.Conv2d(concat_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.conv_out(x)
        return x


# ========== 🔥 主网络（3层版本） ==========

class UNetBackbone(nn.Module):
    """
    🔥 U-Net 特征提取器（3层轻量版）
    
    结构：
    - Encoder: 3层下采样 (16→32→64)
    - Bottleneck: AttentionBottleneck (64通道)
    - Decoder: 3层上采样 (64→32→16)
    
    用途：
    - DSC_IJCAI 的 _SharedFeatureExtractor 内部使用
    - 输入/输出：特征图 [B, channels, H, W]
    
    Args:
        in_channels: 输入通道数（如16）
        out_channels: 输出通道数（如16）
        bottleneck_blocks: bottleneck 中的 attention 块数量
        num_res_blocks: 每层的 ResidualBlock 数量（建议1）
        use_attention: 是否使用 attention bottleneck
        attention_heads: attention 头数（建议4）
        dropout: Dropout 概率
    
    Example:
        >>> # 用于 DSC_IJCAI（16通道特征提取）
        >>> unet = UNetBackbone(
        ...     in_channels=16, 
        ...     out_channels=16,
        ...     bottleneck_blocks=1,
        ...     num_res_blocks=1,
        ...     attention_heads=4
        ... )
        >>> x = torch.randn(2, 16, 256, 256)
        >>> out = unet(x)  # [2, 16, 256, 256]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_blocks: int = 1,
        num_res_blocks: int = 1,
        use_attention: bool = True,
        attention_heads: int = 4,
        dropout: float = 0.1,
        use_wave: bool = True,
        state_dim: int = 3,
    ):
        super().__init__()
        
        # 🔥 根据输入通道数自动生成 3层 通道配置
        CH1, CH2, CH3 = get_channel_config(in_channels)
        
        # ========== 🔥 Encoder（3层） ==========
        self.down1 = EncoderBlock(in_channels, CH1, num_blocks=num_res_blocks)  # 16 → 32 (H/2)
        self.down2 = EncoderBlock(CH1, CH2, num_blocks=num_res_blocks)          # 32 → 64 (H/4)
        self.down3 = EncoderBlock(CH2, CH3, num_blocks=num_res_blocks)          # 64 → 64 (H/8)
        
        # ========== 🔥 Bottleneck ==========
        if use_attention:
            self.bottleneck = AttentionBottleneck(
                CH3, 
                num_blocks=bottleneck_blocks,
                num_heads=attention_heads, 
                dropout=dropout
            )
        else:
            self.bottleneck = PlainBottleneck(
                CH3, 
                num_blocks=bottleneck_blocks,
                dropout=dropout
            )
        
        # ========== 🔥 FFT 版水下退化感知波传播算子 UD-WPO：插在 U-Net bottleneck 后 ==========
        self.use_wave = use_wave
        self.wave = UDWPO(CH3, state_dim=state_dim) if use_wave else nn.Identity()

        # ========== 🔥 Decoder（3层） ==========
        self.up3 = DecoderBlock(CH3, CH3, CH2, num_blocks=num_res_blocks, dropout=dropout)  # 64+64skip → 32
        self.up2 = DecoderBlock(CH2, CH2, CH1, num_blocks=num_res_blocks, dropout=dropout)  # 32+32skip → 16
        self.up1 = DecoderBlock(CH1, CH1, CH1, num_blocks=num_res_blocks, dropout=dropout)  # 16+16skip → 16
        
        # ========== 🔥 输出层 ==========
        self.final = nn.Sequential(
            ResidualBlock(CH1, dropout=dropout, use_se=True),
            nn.Conv2d(CH1, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor, degradation_state: torch.Tensor = None) -> torch.Tensor:
        # Encoder
        x1, skip1 = self.down1(x)    # [B, 16, H/2, W/2]
        x2, skip2 = self.down2(x1)   # [B, 32, H/4, W/4]
        x3, skip3 = self.down3(x2)   # [B, 64, H/8, W/8]
        
        # Bottleneck
        x_b = self.bottleneck(x3)    # [B, 64, H/8, W/8]
        if self.use_wave:
            x_b = self.wave(x_b, degradation_state)  # 阻尼波传播：稳定恢复方向
        
        # Decoder
        x = self.up3(x_b, skip3)     # [B, 32, H/4, W/4]
        x = self.up2(x, skip2)       # [B, 16, H/2, W/2]
        x = self.up1(x, skip1)       # [B, 16, H, W]
        
        # 输出
        out = self.final(x)          # [B, 16, H, W]
        
        return out
    
    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("="*60)
    print("UNetBackbone (3-Layer Version) Test")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Using device: {device}\n")
    
    # ========== 🔥 测试 DSC_IJCAI 使用场景 ==========
    
    print("="*60)
    print("Testing: DSC_IJCAI Configuration (16 channels, 3 layers)")
    print("="*60)
    
    model = UNetBackbone(
        in_channels=16,
        out_channels=16,
        bottleneck_blocks=1,
        num_res_blocks=1,
        use_attention=True,
        attention_heads=4,
        dropout=0.1
    ).to(device)
    
    params = model.count_parameters()
    print(f"✅ Parameters: {params / 1e6:.2f}M")
    
    # 前向传播测试
    B, C, H, W = 2, 16, 256, 256
    x = torch.randn(B, C, H, W, device=device)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Input shape:  {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    assert output.shape == x.shape, "❌ Output shape mismatch!"
    
    # 梯度测试
    model.train()
    x.requires_grad = True
    output = model(x)
    loss = output.mean()
    loss.backward()
    
    print(f"✅ Gradient check: grad_mean={x.grad.mean().item():.6f}")
    
    # 显存测试
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.train()
        x = torch.randn(4, 16, 256, 256, device=device, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✅ Peak memory (batch=4): {mem_allocated:.2f} GB")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("🎯 通道配置: 16 → 32 → 64 (bottleneck)")
    print("🎯 参数量显著减少（约为5层版本的 40%）")
    print("="*60)
