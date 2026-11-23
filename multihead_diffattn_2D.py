import math
import torch
import torch.nn.functional as F
from torch import nn

from kernel.rotary import apply_rotary_emb
# from flash_attn import flash_attn_func
from kernel.rotary import RotaryEmbedding

from rms_norm import RMSNorm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    对键值张量进行重复操作，用于扩展KV头的数量以匹配查询头数量。

    参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, n_kv_heads, seq_len, head_dim)
        n_rep (int): 重复次数，即每个KV头需要重复多少次以匹配Q头数量

    返回:
        torch.Tensor: 扩展后的张量，形状为 (batch_size, n_kv_heads * n_rep, seq_len, head_dim)
    """
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    """
    根据层深度计算lambda初始值的函数。

    参数:
        depth (int): 当前网络层的深度索引

    返回:
        float: 计算得到的lambda初始值
    """
    # print("初始值")
    # print(0.8 - 0.6 * math.exp(-0.3 * depth))
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RotaryEmbedding2D(nn.Module):
    """
    2D旋转位置编码，适用于图像任务
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D rotary embedding"
        self.dim = dim
        self.height_dim = dim // 2  # height维度的位置编码维度
        self.width_dim = dim // 2   # width维度的位置编码维度
        
        # 创建height和width的位置编码（注意：这些实例当前未使用，可根据需要集成）
        self.height_rotary = RotaryEmbedding(dim=self.height_dim)
        self.width_rotary = RotaryEmbedding(dim=self.width_dim)
    
    def forward(self, x, height, width):
        """
        生成2D位置编码
        
        参数:
            x: 输入张量
            height: 图像高度
            width: 图像宽度
        
        返回:
            cos, sin: 2D位置编码的余弦和正弦分量
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        device = x.device
        
        # 生成位置坐标
        h_pos = torch.arange(height, device=device).unsqueeze(1).expand(height, width).reshape(-1)  # [h*w]
        w_pos = torch.arange(width, device=device).unsqueeze(0).expand(height, width).reshape(-1)   # [h*w]
        
        # 为height和width分别生成位置编码（修正：使用 full height_dim 而非 //2）
        h_cos, h_sin = self._get_1d_encoding(h_pos, self.height_dim, device)  # [h*w, height_dim//2]
        w_cos, w_sin = self._get_1d_encoding(w_pos, self.width_dim, device)   # [h*w, width_dim//2]
        
        # 组合height和width的位置编码
        cos = torch.cat([h_cos, w_cos], dim=-1)  # [h*w, dim//2]
        sin = torch.cat([h_sin, w_sin], dim=-1)  # [h*w, dim//2]
        
        # 扩展到batch维度
        cos = cos.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, h*w, dim//2]
        sin = sin.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, h*w, dim//2]
        
        return cos, sin
    
    def _get_1d_encoding(self, pos, dim, device):
        """
        生成1D位置编码
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        sinusoid_inp = torch.outer(pos.float(), inv_freq)
        cos = torch.cos(sinusoid_inp)
        sin = torch.sin(sinusoid_inp)
        return cos, sin


def apply_rotary_emb_2d(x, cos, sin, interleaved=False):
    """
    应用2D旋转位置编码
    
    参数:
        x: 输入张量 [batch_size, seq_len, num_heads, head_dim]
        cos, sin: 位置编码的余弦和正弦分量
        interleaved: 是否使用交错模式
    """
    if interleaved:
        # 对于差异注意力的交错模式
        x1 = x[..., ::2]   # 偶数索引
        x2 = x[..., 1::2]  # 奇数索引
        
        # 扩展cos和sin到匹配的维度
        cos = cos.unsqueeze(2).expand(-1, -1, x.shape[2], -1)  # [batch_size, seq_len, num_heads, head_dim//2]
        sin = sin.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        
        # 应用旋转
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # 重新组合
        rotated_x = torch.empty_like(x)
        rotated_x[..., ::2] = rotated_x1
        rotated_x[..., 1::2] = rotated_x2
        
        return rotated_x
    else:
        # 标准模式
        seq_len, head_dim = x.shape[1], x.shape[-1]
        half_dim = head_dim // 2
        
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        
        cos = cos.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        sin = sin.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


class MultiheadDiffAttn(nn.Module):
    """
    多头差异注意力机制模块，实现了一种改进的注意力计算方式。

    参数:
        embed_dim (int): 嵌入维度
        depth (int): 当前层索引，用于初始化lambda参数
        num_heads (int): 注意力头的数量
        num_kv_heads (int, optional): 键值头的数量，用于组查询注意力(GQA)。默认为None表示使用多头注意力(MHA)
    """

    def __init__(
        self,
        embed_dim,
        depth, # 当前层索引
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads设置为基准Transformer的num_heads的一半
        # 例如，要与具有16个头的基准Transformer进行比较，请传入num_heads=8用于DIFF Transformer
        self.num_heads = num_heads
        
        # 如果使用GQA，arg num_kv_heads设置为基准Transformer的num_kv_heads的一半
        # 例如，要与具有16个头和8个kv_heads的基准Transformer进行比较，
        # 请传入num_heads=8, num_kv_heads=4用于DIFF Transformer
        # 如果使用MHA，请传入num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.pos_emb_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth表示当前层索引
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        # 使用2D旋转位置编码，适合图像任务
        self.rotary_emb = RotaryEmbedding2D(dim=self.head_dim)

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
        key_padding_mask=None,
    ):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)
            rel_pos: 相对位置信息（未使用，由内部重新计算）
            attn_mask (torch.Tensor, optional): 注意力掩码
            key_padding_mask (torch.Tensor, optional): 键填充掩码

        返回:
            torch.Tensor: 输出张量，形状与输入相同 (batch_size, channels, height, width)
        """
        # 将输入从CNN格式转换为序列格式
        batch_size, c, h, w = x.size()
        x_reshaped = x.view(batch_size, c, h * w)
        x = x_reshaped.transpose(1, 2)
        bsz, tgt_len, embed_dim = x.size()

        src_len = tgt_len

        # 投影QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑QKV张量
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # 应用2D旋转位置编码
        cos, sin = self.rotary_emb(x, h, w)
        
        q = apply_rotary_emb_2d(q, cos, sin, interleaved=True)
        k = apply_rotary_emb_2d(k, cos, sin, interleaved=True)

        # 调整张量维度并计算注意力权重
        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        
        # 对于2D图像，通常不需要因果掩码，但保留这个选项
        if attn_mask is None:
            # 对于图像任务，可以选择不使用因果掩码，或者使用适合2D的掩码
            # 这里保持原来的逻辑，但在图像任务中通常attn_mask会被显式设置
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        
        attn_weights = torch.nan_to_num(attn_weights)
        if attn_mask is not None:
            attn_weights += attn_mask   
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            # key_padding_mask形状: (bsz, src_len)
            # 转换为 (bsz, 1, 1, src_len) 并扩展
            mask = key_padding_mask.view(bsz, 1, 1, -1)
            mask = mask.expand(-1, self.num_heads * 2, tgt_len, -1).to(attn_weights.device)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
            
        # softmax归一化
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        # 计算lambda参数并应用差异注意力
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        # 计算最终注意力输出
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)

        # 将输出从序列格式转换回CNN格式
        bsz, tgt_len, embed_dim = attn.size()
        attn_transposed = attn.transpose(1, 2)  # 恢复为 [batch_size, c, h*w]
        attn_original = attn_transposed.view(bsz, embed_dim, h, w)  # 恢复原始形状 [batch_size, c, h, w]
        #print("attn_original:", attn_original.shape)
        return attn_original