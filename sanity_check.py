"""Sanity check: forward pass with 2-view config on CPU/MPS (mac M4).

Run:  python sanity_check.py --num_view 2 --frames 27 --depth 2 --model dsvtformer
"""
import sys
import torch

from common.opt import opts

opt = opts().parse()
exec('from model.' + opt.model + ' import Model')


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    device = pick_device()
    print(f'[sanity] device = {device}')
    print(f'[sanity] num_view = {opt.num_view}, frames = {opt.frames}, depth = {opt.depth}')

    B = 2
    F = opt.frames
    V = opt.num_view
    J = 17
    IMG_FEAT = 256

    model = Model(
        num_frame=F, num_joints=J, in_chans=2,
        embed_dim_ratio=opt.embed_dim_ratio,
        img_embed_dim_ratio=opt.img_embed_dim_ratio,
        depth=opt.depth, num_heads=8, mlp_ratio=2.,
        qkv_bias=True, qk_scale=None, drop_path_rate=0.,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'[sanity] params = {n_params:.2f} M')

    x = torch.randn(B, F, V, J, 2, device=device)
    img = torch.randn(B, F, V, IMG_FEAT, device=device)
    print(f'[sanity] input  x.shape   = {tuple(x.shape)}')
    print(f'[sanity] input  img.shape = {tuple(img.shape)}')

    model.eval()
    with torch.no_grad():
        y = model(x, img)
    print(f'[sanity] output y.shape   = {tuple(y.shape)}')

    expected = (B, F, J, 3)
    assert tuple(y.shape) == expected, f'Shape mismatch: got {tuple(y.shape)}, expected {expected}'
    print('[sanity] OK — forward pass passed for 2-view config')


if __name__ == '__main__':
    main()
