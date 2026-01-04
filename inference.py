"""Inference + skeleton animation for DSVTformer (2-view).

Usage:
    python inference.py \\
        --previous_dir checkpoint/best.pth \\
        --num_view 2 --view_indices 0,2 \\
        --frames 27 --depth 2 --model dsvtformer \\
        --out_dir ./outputs

Inputs:
    - 2D keypoints + image features for 2 views (npz/pkl, same format as training)
    - Trained checkpoint via --previous_dir

Outputs:
    - outputs/pred_3d.npy           : (T, J, 3) predicted 3D poses
    - outputs/pose_animation.mp4    : 3D skeleton animation
"""
import os
import argparse

import numpy as np
import torch

from common.opt import opts

opt = opts().parse()
exec('from model.' + opt.model + ' import Model')


H36M_BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
    (11, 14),
]
LEFT_JOINTS = {4, 5, 6, 11, 12, 13}
RIGHT_JOINTS = {1, 2, 3, 14, 15, 16}


def pick_device():
    if torch.cuda.is_available():
        return torch.device(f'cuda:{opt.gpu}')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_inputs(path_2d, path_img):
    """Load 2D keypoints and image features.

    Expected shapes:
        kpts:  (T, V, J, 2)  — V must equal opt.num_view
        feats: (T, V, 256)
    """
    kpts = np.load(path_2d)
    feats = np.load(path_img)
    assert kpts.ndim == 4 and kpts.shape[1] == opt.num_view, \
        f'2D keypoints must be (T, {opt.num_view}, J, 2), got {kpts.shape}'
    assert feats.ndim == 3 and feats.shape[1] == opt.num_view, \
        f'image features must be (T, {opt.num_view}, 256), got {feats.shape}'
    return kpts.astype(np.float32), feats.astype(np.float32)


def slide_windows(arr, frames):
    """Tile a (T, ...) array into overlapping windows of length `frames` centered on each frame."""
    pad = (frames - 1) // 2
    padded = np.pad(arr, [(pad, pad)] + [(0, 0)] * (arr.ndim - 1), mode='edge')
    return np.stack([padded[i:i + frames] for i in range(arr.shape[0])], axis=0)


@torch.no_grad()
def run_inference(model, kpts, feats, device, batch=32):
    F = opt.frames
    pad = (F - 1) // 2
    kpts_w = slide_windows(kpts, F)   # (T, F, V, J, 2)
    feats_w = slide_windows(feats, F)  # (T, F, V, 256)

    preds = []
    for i in range(0, kpts_w.shape[0], batch):
        x = torch.from_numpy(kpts_w[i:i + batch]).to(device)
        m = torch.from_numpy(feats_w[i:i + batch]).to(device)
        y = model(x, m)                    # (B, F, J, 3)
        preds.append(y[:, pad].cpu().numpy())  # take center frame
    return np.concatenate(preds, axis=0)   # (T, J, 3)


def render_animation(poses_3d, out_path, fps=25):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    # H36M camera frame: X right, Y down, Z forward.
    # Matplotlib 3D: vertical axis is Z. Remap so that "up" = -Y_cam.
    #   plot_X = cam_X        (left/right)
    #   plot_Y = cam_Z        (depth into scene)
    #   plot_Z = -cam_Y       (up)
    poses_3d = np.stack([poses_3d[..., 0], poses_3d[..., 2], -poses_3d[..., 1]], axis=-1)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Per-frame extent so the axes follow the subject (camera-tracking style).
    spans = poses_3d.max(axis=1) - poses_3d.min(axis=1)   # (T, 3)
    rng = spans.max() / 2 * 1.2

    def draw(t):
        ax.cla()
        p = poses_3d[t]
        c = p[0]  # hip/root — recenter axes each frame
        ax.set_xlim(c[0] - rng, c[0] + rng)
        ax.set_ylim(c[1] - rng, c[1] + rng)
        ax.set_zlim(c[2] - rng, c[2] + rng)
        ax.set_title(f'frame {t}')
        for a, b in H36M_BONES:
            if a in LEFT_JOINTS or b in LEFT_JOINTS:
                color = 'tab:blue'
            elif a in RIGHT_JOINTS or b in RIGHT_JOINTS:
                color = 'tab:red'
            else:
                color = 'black'
            ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], [p[a, 2], p[b, 2]], color=color, lw=2)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=10, c='k')

    anim = FuncAnimation(fig, draw, frames=poses_3d.shape[0], interval=1000 / fps)

    if out_path.endswith('.mp4'):
        try:
            anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2000))
        except Exception as e:
            print(f'[anim] FFMpeg unavailable ({e}); falling back to .gif')
            out_path = out_path.replace('.mp4', '.gif')
            anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'[anim] saved -> {out_path}')


def main():
    extra = argparse.ArgumentParser()
    extra.add_argument('--kpts_2d', type=str, required=True, help='path to (T,V,J,2) .npy')
    extra.add_argument('--img_feat', type=str, required=True, help='path to (T,V,256) .npy')
    extra.add_argument('--out_dir', type=str, default='./outputs')
    extra.add_argument('--fps', type=int, default=25)
    args, _ = extra.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device()
    print(f'[infer] device = {device}, num_view = {opt.num_view}, frames = {opt.frames}')

    kpts, feats = load_inputs(args.kpts_2d, args.img_feat)
    print(f'[infer] kpts {kpts.shape}, feats {feats.shape}')

    model = Model(
        num_frame=opt.frames, num_joints=17, in_chans=2,
        embed_dim_ratio=opt.embed_dim_ratio,
        img_embed_dim_ratio=opt.img_embed_dim_ratio,
        depth=opt.depth, num_heads=8, mlp_ratio=2.,
        qkv_bias=True, qk_scale=None, drop_path_rate=0.,
    ).to(device)

    assert opt.previous_dir, 'pass --previous_dir <checkpoint.pth>'
    state = torch.load(opt.previous_dir, map_location=device)
    model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()})
    model.eval()

    poses = run_inference(model, kpts, feats, device, batch=32)
    out_npy = os.path.join(args.out_dir, 'pred_3d.npy')
    np.save(out_npy, poses)
    print(f'[infer] saved -> {out_npy}, shape {poses.shape}')

    render_animation(poses, os.path.join(args.out_dir, 'pose_animation.mp4'), fps=args.fps)


if __name__ == '__main__':
    main()
