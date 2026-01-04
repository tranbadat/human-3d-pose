"""End-to-end demo: arbitrary single-view video -> DSVTformer -> 3D animation.

Pipeline:
    1. Run MediaPipe Pose on every frame to get 33 body landmarks.
    2. Remap to H36M 17-joint topology.
    3. Synthesize a 2nd view by horizontal mirror (since DSVTformer needs V=2).
       *** This is NOT a real second camera — depth output will be approximate. ***
    4. Build surrogate 256-D image features (no raw image branch available).
    5. Run DSVTformer inference.
    6. Render side-by-side: original video + predicted 3D skeleton.

Usage:
    python scripts/video_to_3d.py --video testvideo.mp4 \
        --ckpt outputs_h3wb/mini_best.pth --out_dir outputs_video
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pre-set sys.argv for opt parser then strip our extras
import argparse as _ap
_extra = _ap.ArgumentParser(add_help=False)
_extra.add_argument('--video', required=True)
_extra.add_argument('--ckpt', required=True)
_extra.add_argument('--out_dir', default='./outputs_video')
_extra.add_argument('--max_frames', type=int, default=240)
_extra_args, _ = _extra.parse_known_args()

from common.opt import opts
opt = opts().parse()
exec('from model.' + opt.model + ' import Model')


def pick_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def coco17_to_h36m(coco):
    """Map COCO 17 keypoints (YOLOv8 pose order) to H36M 17 joints."""
    out = np.zeros((17, 2), dtype=np.float32)
    pelvis = (coco[11] + coco[12]) / 2
    thorax = (coco[5] + coco[6]) / 2
    out[0] = pelvis
    out[1] = coco[12]; out[2] = coco[14]; out[3] = coco[16]   # right leg
    out[4] = coco[11]; out[5] = coco[13]; out[6] = coco[15]   # left leg
    out[7] = (pelvis + thorax) / 2
    out[8] = thorax
    out[9] = coco[0]; out[10] = coco[0]                        # nose -> neck/head
    out[11] = coco[5]; out[12] = coco[7]; out[13] = coco[9]    # left arm
    out[14] = coco[6]; out[15] = coco[8]; out[16] = coco[10]   # right arm
    return out


def extract_2d(video_path, max_frames):
    from ultralytics import YOLO
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_bgr, kpts = [], []
    while True:
        ok, frame = cap.read()
        if not ok or len(frames_bgr) >= max_frames: break
        res = model(frame, verbose=False)[0]
        if res.keypoints is None or res.keypoints.xyn.shape[0] == 0:
            if not kpts: continue
            kpts.append(kpts[-1])
        else:
            # pick largest-bbox detection
            xyn = res.keypoints.xyn.cpu().numpy()  # (N, 17, 2) normalized
            if res.boxes is not None and len(res.boxes) > 1:
                areas = res.boxes.xywh.cpu().numpy()[:, 2] * res.boxes.xywh.cpu().numpy()[:, 3]
                idx = int(areas.argmax())
            else:
                idx = 0
            kpts.append(coco17_to_h36m(xyn[idx]))
        frames_bgr.append(frame)
    cap.release()
    return np.stack(kpts), frames_bgr, fps, (W, H)


def make_two_views(kpts_h36m):
    """Real view + horizontally-mirrored view (fake 2nd camera).

    Disclaimer: this gives the model two correlated 2D inputs but no real
    parallax, so 3D depth is estimated mostly from the model's pose prior.
    """
    v0 = kpts_h36m.copy()                    # (T, 17, 2) in [0,1]
    v1 = v0.copy(); v1[..., 0] = 1.0 - v1[..., 0]
    # Center & scale to ~[-1, 1] like the H3WB normalization in mini-train
    both = np.stack([v0, v1], axis=1)        # (T, 2, 17, 2)
    c = both.reshape(-1, 2).mean(0)
    s = max(np.abs(both - c).max(), 1.0)
    return ((both - c) / s).astype(np.float32)


def slide_windows(arr, frames):
    pad = (frames - 1) // 2
    padded = np.pad(arr, [(pad, pad)] + [(0, 0)] * (arr.ndim - 1), mode='edge')
    return np.stack([padded[i:i + frames] for i in range(arr.shape[0])], axis=0)


@torch.no_grad()
def run_inference(model, kpts2d, device, frames, batch=16):
    F, V, J = frames, kpts2d.shape[1], kpts2d.shape[2]
    win = slide_windows(kpts2d, F)            # (T, F, V, J, 2)
    rng = np.random.default_rng(0)
    proj = rng.standard_normal((J * 2, 256)).astype(np.float32) * 0.05
    feat = np.tanh(win.reshape(*win.shape[:3], J * 2) @ proj).astype(np.float32)
    pad = (F - 1) // 2
    out = []
    for i in range(0, win.shape[0], batch):
        x = torch.from_numpy(win[i:i + batch]).to(device)
        m = torch.from_numpy(feat[i:i + batch]).to(device)
        y = model(x, m)
        out.append(y[:, pad].cpu().numpy())
    return np.concatenate(out, axis=0)        # (T, 17, 3)


H36M_BONES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
              (8,11),(11,12),(12,13),(8,14),(14,15),(15,16),
              (11,14)]
LEFT = {4,5,6,11,12,13}; RIGHT = {1,2,3,14,15,16}


def render_side_by_side(frames_bgr, poses_3d, out_path, fps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    # H36M cam frame -> matplotlib up: (X, Z, -Y)
    p = np.stack([poses_3d[..., 0], poses_3d[..., 2], -poses_3d[..., 1]], axis=-1)
    spans = p.max(axis=1) - p.min(axis=1)
    rng_ = spans.max() / 2 * 1.2

    fig = plt.figure(figsize=(10, 5))
    ax_v = fig.add_subplot(1, 2, 1)
    ax_3 = fig.add_subplot(1, 2, 2, projection='3d')

    def draw(t):
        ax_v.cla(); ax_3.cla()
        ax_v.imshow(cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2RGB))
        ax_v.set_title(f'input frame {t}'); ax_v.axis('off')
        pt = p[t]
        c = pt[0]
        ax_3.set_xlim(c[0]-rng_, c[0]+rng_)
        ax_3.set_ylim(c[1]-rng_, c[1]+rng_)
        ax_3.set_zlim(c[2]-rng_, c[2]+rng_)
        ax_3.set_title('predicted 3D skeleton')
        for a, b in H36M_BONES:
            c = 'tab:blue' if a in LEFT or b in LEFT else ('tab:red' if a in RIGHT or b in RIGHT else 'black')
            ax_3.plot([pt[a,0], pt[b,0]], [pt[a,1], pt[b,1]], [pt[a,2], pt[b,2]], color=c, lw=2)
        ax_3.scatter(pt[:,0], pt[:,1], pt[:,2], s=10, c='k')

    n = min(len(frames_bgr), p.shape[0])
    anim = FuncAnimation(fig, draw, frames=n, interval=1000 / fps)
    if out_path.endswith('.mp4'):
        try: anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2000))
        except Exception as e:
            print(f'[anim] FFMpeg unavailable ({e}); -> .gif')
            out_path = out_path.replace('.mp4', '.gif')
            anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'[anim] saved -> {out_path}')


def main():
    args = _extra_args
    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device()
    print(f'[video2 3d] device={device}  video={args.video}  ckpt={args.ckpt}')

    kpts, frames_bgr, fps, wh = extract_2d(args.video, args.max_frames)
    print(f'[video2 3d] extracted {kpts.shape[0]} pose frames at {fps:.1f} fps, res={wh}')
    if kpts.shape[0] < opt.frames:
        raise SystemExit(f'video too short: need >= {opt.frames} frames')

    kpts2v = make_two_views(kpts)             # (T, 2, 17, 2) normalized
    print(f'[video2 3d] 2-view tensor: {kpts2v.shape}  (view 1 = horizontal mirror surrogate)')

    model = Model(num_frame=opt.frames, num_joints=17, in_chans=2,
                  embed_dim_ratio=opt.embed_dim_ratio,
                  img_embed_dim_ratio=opt.img_embed_dim_ratio,
                  depth=opt.depth, num_heads=8, mlp_ratio=2.,
                  qkv_bias=True, qk_scale=None, drop_path_rate=0.).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()})
    model.eval()

    poses = run_inference(model, kpts2v, device, opt.frames)
    np.save(os.path.join(args.out_dir, 'pred_3d.npy'), poses)
    print(f'[video2 3d] poses {poses.shape} saved')
    render_side_by_side(frames_bgr, poses, os.path.join(args.out_dir, 'demo.mp4'), fps)


if __name__ == '__main__':
    main()
