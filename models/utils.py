import torch
import torch.nn.functional as F


def depth_normalization(depth_map, eps=1.0):
    """
    Depth normalization for batched depth maps with mask (0 = invalid depth).

    Args:
        depth_map (torch.Tensor): shape [B, 1, H, W], 0 means invalid depth
        eps (float): small constant for numerical stability

    Returns:
        torch.Tensor: normalized depth map, same shape as input
    """
    # 1. Create mask: valid depth > 0
    mask = (depth_map > 0).float()  # [B, 1, H, W]

    # 2. log-scale transform (only for valid pixels)
    # d_tilde = torch.log(depth_map + eps) * mask  # avoid log(0)
    d_tilde = depth_map
    B = d_tilde.shape[0]

    # 3. Flatten valid pixels per batch
    d_tilde_flat = d_tilde.view(B, -1)
    mask_flat = mask.view(B, -1)

    # Replace invalid entries with +inf for min, -inf for max exclusion
    d_valid = torch.where(mask_flat.bool(), d_tilde_flat, torch.tensor(float('nan'), device=depth_map.device))

    # 4. Compute per-sample percentiles on valid pixels
    d_min = torch.nanquantile(d_valid, 0.0, dim=1, keepdim=True)  # [B, 1]
    d_max = torch.nanquantile(d_valid, 1.0, dim=1, keepdim=True)  # [B, 1]

    # Reshape for broadcasting
    d_min = d_min.view(B, 1, 1, 1)
    d_max = d_max.view(B, 1, 1, 1)

    # 5. Normalize (only valid pixels)
    d_hat = torch.zeros_like(depth_map)
    valid_range = (d_max - d_min + 1e-6)
    d_hat[mask.bool()] = ((d_tilde[mask.bool()] - d_min.view(B, 1)[0]) / valid_range.view(B, 1)[0])

    # 6. Keep invalid values = 0
    d_hat = d_hat * mask

    return d_hat


def interpolate_depth_nearest_zero_invalid(depth, size=None, scale_factor=None):
    """
    对0表示无效的深度图进行最近邻插值（仅扩展有效区域）
    Args:
        depth: [B,1,H,W]，0为无效
    Returns:
        depth_up: 插值后的深度图，0仍代表无效
    """
    mask = (depth > 0).float()

    # 先插值深度 × 掩码
    v_up = F.interpolate(depth * mask, size=size, scale_factor=scale_factor, mode='nearest')
    m_up = F.interpolate(mask, size=size, scale_factor=scale_factor, mode='nearest')

    # 保持无效区域为0
    depth_up = torch.where(m_up > 0, v_up, torch.zeros_like(v_up))
    return depth_up

import torch
import torch.nn.functional as F

@torch.no_grad()
def fit_scale_shift_batch(D_pred, D_tof, eps: float = 1e-6):
    """
    在 batch 内对每个样本独立拟合 scale/shift:
        depth_aligned = s * D_pred + t
    其中有效像素由 D_tof > 0 决定。
    Args:
        D_pred: [B,1,H,W]  预测深度（>0）
        D_tof : [B,1,H,W]  ToF深度（0 表示无效）
        eps   : 数值稳定项
    Returns:
        depth_aligned: [B,1,H,W]
        s: [B,1,1,1]
        t: [B,1,1,1]
    """
    assert D_pred.dim() == 4 and D_tof.dim() == 4 and D_pred.shape == D_tof.shape
    B = D_pred.shape[0]

    # 有效权重（这里就是有效掩码），形状 [B,1,H,W]
    w = (D_tof > 0).to(D_pred.dtype)

    p = D_pred
    g = D_tof

    # 逐样本汇总（在有效像素上）
    dims = (1, 2, 3)
    Sw   = (w).sum(dim=dims)                 # [B]
    Swp  = (w * p).sum(dim=dims)             # [B]
    Swg  = (w * g).sum(dim=dims)             # [B]
    Swp2 = (w * p * p).sum(dim=dims)         # [B]
    Swpg = (w * p * g).sum(dim=dims)         # [B]

    Delta = Swp2 * Sw - Swp * Swp            # [B]

    s = torch.zeros_like(Sw)
    t = torch.zeros_like(Sw)

    valid = (Delta > 0)  # 有效像素非空的样本

    s[valid] = (Sw[valid] * Swpg[valid] - Swp[valid] * Swg[valid]) / Delta[valid]          # [B]
    t[valid] = (Swp2[valid] * Swg[valid] - Swp[valid] * Swpg[valid]) / Delta[valid]      # [B]

    # 组装输出
    s_view = s.view(B, 1, 1, 1)
    t_view = t.view(B, 1, 1, 1)
    depth_aligned = s_view * D_pred + t_view  # [B,1,H,W]

    return depth_aligned, s_view, t_view


@torch.no_grad()
def fit_scale_shift_batch_conf(
    D_pred, D_tof, confidence=None,
    eps: float = 1e-12, use_float64: bool = False
):
    """
    直接解：
        min_{a,b} sum_i w_i * (a*D_pred_i + b - D_tof_i)^2
    其中 w_i = confidence_i * 1[D_tof_i>0]（若不给 confidence，则仅用有效掩码）

    Args:
        D_pred      : [B,1,H,W]  预测深度
        D_tof       : [B,1,H,W]  ToF深度（0=无效）
        confidence  : [B,1,H,W]  像素置信度 in [0,1]，可为 None
        eps  : 数值稳定参数
        use_float64 : 统计量累加是否使用更高精度（AMP/FP16 时建议 True）
    Returns:
        D_aligned: [B,1,H,W]  = a*D_pred + b
        a: [B,1,1,1]  缩放
        b: [B,1,1,1]  平移
    """
    assert D_pred.shape == D_tof.shape and D_pred.dim() == 4
    B = D_pred.shape[0]

    acc_dtype = torch.float64 if use_float64 else torch.float32
    x = D_pred.to(acc_dtype)
    y = D_tof.to(acc_dtype)

    # 权重：ToF 有效掩码 * 置信度（若无置信度则=掩码）
    mask = (y > 0).to(acc_dtype)
    if confidence is not None:
        w = (confidence * mask).to(acc_dtype)
    else:
        w = mask

    dims = (1, 2, 3)
    # 加权统计量（与推导中的记号一致）
    Sw   = (w).sum(dim=dims)                  # [B]
    Swx  = (w * x).sum(dim=dims)              # [B]
    Swy  = (w * y).sum(dim=dims)              # [B]
    Swx2 = (w * x * x).sum(dim=dims)          # [B]
    Swxy = (w * x * y).sum(dim=dims)          # [B]

    # 法方程矩阵：
    # [ Swx2   Swx ] [a] = [ Swxy ]
    # [ Swx     Sw ] [b]   [ Swy  ]
    Delta = Swx2 * Sw - (Swx * Swx)    # [B]

    # 闭式解（不中心化）
    a = (Sw * Swxy - Swx * Swy) / (Delta + eps)  # [B]
    b = (Swx2 * Swy - Swx * Swxy) / (Delta + eps)

    # 退化处理：Delta 过小（p 近常数），只估计平移 t；Sw==0（无有效像素）→ s=t=0
    degenerate = (Delta.abs() < eps)
    no_valid   = (Sw <= 0)

    t_only = Swy / (Sw + eps)                            # [B]
    s = torch.where(degenerate, torch.zeros_like(s), s)
    t = torch.where(degenerate, t_only, t)

    s = torch.where(no_valid, torch.zeros_like(s), s)
    t = torch.where(no_valid, torch.zeros_like(t), t)

    # 组装输出
    a_view = a.view(B, 1, 1, 1).to(D_pred.dtype)
    b_view = b.view(B, 1, 1, 1).to(D_pred.dtype)
    D_aligned = a_view * D_pred + b_view

    return D_aligned, a_view, b_view
