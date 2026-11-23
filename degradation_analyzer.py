import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationAnalyzer(nn.Module):
    """
    可学习三维水下退化状态估计器。

    输入:
        img: [B, 3, H, W], 范围 [0, 1]

    输出:
        deg_vec: [B, 3]
            deg_vec[:, 0] = scattering strength，散射强度
            deg_vec[:, 1] = color cast strength，偏色程度
            deg_vec[:, 2] = low-light strength，弱光程度

    注意:
        这里保留类名 DegradationAnalyzer，是为了不改 DSC_ICME_NAS.py 里的 import。
        但它的含义已经从“12维手工统计特征”改成“可学习3维退化状态”。
    """

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        feat = self.encoder(img)
        feat = feat.flatten(1)

        deg_vec = self.fc(feat)
        deg_vec = torch.nan_to_num(deg_vec, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        return deg_vec


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DegradationAnalyzer().to(device)
    x = torch.rand(2, 3, 256, 256).to(device)
    y = model(x)
    print("Input:", x.shape)
    print("Deg vec:", y.shape, y)
    assert y.shape == (2, 3)
    assert torch.isfinite(y).all()