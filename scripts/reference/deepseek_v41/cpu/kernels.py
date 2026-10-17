"""CPU arithmetic, with explicit BF16 rounding at reference kernel boundaries."""

import torch

FP4 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def unpack_fp4(x):
    x = x.view(torch.uint8)
    codes = torch.stack((x & 15, x >> 4), -1).flatten(-2).long()
    return torch.tensor(FP4)[codes]


def round_scale_up(x):
    """Pinned fast_pow2(fast_log2_ceil(x)), on positive normal FP32 values."""
    bits = x.float().contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    return (exponent << 23).view(torch.float32)


def act_quant(
    x, block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu, inplace=False
):
    z = x.float().unflatten(-1, (-1, block_size))
    scale = z.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    if scale_fmt is not None:
        scale = round_scale_up(scale)
    q = (z / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * scale.unsqueeze(-1)).flatten(-2))
        return x
    return q.flatten(-2), scale.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    z = x.float().unflatten(-1, (-1, block_size))
    if scale_dtype == torch.float8_e4m3fn:
        s = (z.abs().amax(-1).clamp_min(6 * 2**-9) / 6).to(scale_dtype).float()
    else:
        s = round_scale_up(z.abs().amax(-1).clamp_min(6 * 2**-126) * (1.0 / 6.0))
    y = (z / s.unsqueeze(-1)).clamp(-6, 6)
    levels = torch.tensor(FP4[:8])
    distance = (y.abs().unsqueeze(-1) - levels).abs()
    # E2M1 ties round to even significand (even code), including subnormals.
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    code = order[distance[..., order].argmin(-1)]
    code = code + torch.signbit(y).long() * 8
    if inplace:
        x.copy_((torch.tensor(FP4)[code] * s.unsqueeze(-1)).flatten(-2))
        return x
    code = code.flatten(-2).to(torch.uint8)
    return code[..., ::2] | (code[..., 1::2] << 4), s.to(scale_dtype)


def fp8_gemm(a, a_scale, b, b_scale, scale_dtype=None, block_size=32):
    a = a.float() * a_scale.float().repeat_interleave(block_size, -1)
    b = (
        b.float()
        * b_scale.float()
        .repeat_interleave(block_size, 0)
        .repeat_interleave(block_size, 1)[: b.shape[0], : b.shape[1]]
    )
    return (a @ b.T).bfloat16()


def fp4_gemm(a, a_scale, b, b_scale, scale_dtype=None, act_block_size=32):
    a = a.float() * a_scale.float().repeat_interleave(act_block_size, -1)
    b = unpack_fp4(b) * b_scale.float().repeat_interleave(32, -1)
    return (a @ b.T).bfloat16()


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    h = hc_mult
    pre = torch.sigmoid(mixes[..., :h] * hc_scale[0] + hc_base[:h]) + eps
    post = 2 * torch.sigmoid(mixes[..., h : 2 * h] * hc_scale[1] + hc_base[h : 2 * h])
    comb = (mixes[..., 2 * h :] * hc_scale[2] + hc_base[2 * h :]).unflatten(
        -1, (h, h)
    ).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    out = torch.empty_like(q)
    for b in range(q.shape[0]):
        for t in range(q.shape[1]):
            maximum = torch.full((q.shape[2],), -1e30)
            denom = torch.zeros_like(maximum)
            acc = torch.zeros(q.shape[2:], dtype=torch.float32)
            for ids in topk_idxs[b, t].split(64):
                valid = ids >= 0
                if not valid.any():
                    continue
                values = kv[b, ids[valid].long()].float()
                scores = q[b, t].float() @ values.T * softmax_scale
                new_max = torch.maximum(maximum, scores.amax(-1))
                rescale = (maximum - new_max).exp()
                probs = (scores - new_max[:, None]).exp()
                denom = denom * rescale + probs.sum(-1)
                if q.dtype == torch.bfloat16:
                    probs = probs.bfloat16().float()
                acc = acc * rescale[:, None] + probs @ values
                maximum = new_max
            denom += (attn_sink - maximum).exp()
            out[b, t] = acc / denom[:, None]
    return out
