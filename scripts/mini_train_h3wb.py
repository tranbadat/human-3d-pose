"""Mini-train DSVTformer on REAL Human3.6M data via the H3WB release.

Source: https://github.com/wholebody3d/wholebody3d  (VideoPose3D-style format,
file 1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC). Contains 4 H36M cameras + 2D/3D
keypoints for S1/S5/S6/S7. We take the first 17 of the 133 COCO-WholeBody
joints and remap them to H36M's 17-joint layout.

Outputs (under outputs_h3wb/):
    - loss_curve.png
    - mini_best.pth
    - kpts_2d.npy / img_feat.npy / gt_3d.npy   (test sequence for inference.py)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.opt import opts
opt = opts().parse()
exec('from model.' + opt.model + ' import Model')

DATA_FILE = './dataset/train_data.npy'
OUT_DIR = './outputs_h3wb'
os.makedirs(OUT_DIR, exist_ok=True)


def coco17_to_h36m17(coco):
    """Map (..., 17, C) COCO body keypoints to H36M 17-joint topology.

    COCO order: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
                9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles.
    """
    out = np.zeros_like(coco)
    out[..., 0, :] = (coco[..., 11, :] + coco[..., 12, :]) / 2     # pelvis
    out[..., 1, :] = coco[..., 12, :]                              # r-hip
    out[..., 2, :] = coco[..., 14, :]                              # r-knee
    out[..., 3, :] = coco[..., 16, :]                              # r-ankle
    out[..., 4, :] = coco[..., 11, :]                              # l-hip
    out[..., 5, :] = coco[..., 13, :]                              # l-knee
    out[..., 6, :] = coco[..., 15, :]                              # l-ankle
    thorax = (coco[..., 5, :] + coco[..., 6, :]) / 2
    pelvis = out[..., 0, :]
    out[..., 7, :] = (pelvis + thorax) / 2                         # spine
    out[..., 8, :] = thorax                                        # thorax
    out[..., 9, :] = coco[..., 0, :]                               # neck/nose
    out[..., 10, :] = coco[..., 0, :]                              # head (approx)
    out[..., 11, :] = coco[..., 5, :]                              # l-shoulder
    out[..., 12, :] = coco[..., 7, :]                              # l-elbow
    out[..., 13, :] = coco[..., 9, :]                              # l-wrist
    out[..., 14, :] = coco[..., 6, :]                              # r-shoulder
    out[..., 15, :] = coco[..., 8, :]                              # r-elbow
    out[..., 16, :] = coco[..., 10, :]                             # r-wrist
    return out


def load_h3wb(data_file, view_indices, frames):
    raw = np.load(data_file, allow_pickle=True).item()
    subjects = sorted(raw.keys())
    print(f'[h3wb] subjects = {subjects}')

    clips_2d, clips_3d = [], []
    for sub in subjects:
        for act, blk in raw[sub].items():
            cams = [c for c in blk.keys() if c not in ('global_3d', 'frame_id')]
            cams = sorted(cams)
            sel = [cams[i] for i in view_indices if i < len(cams)]
            if len(sel) != len(view_indices):
                continue
            T = blk[sel[0]]['pose_2d'].shape[0]
            if T < frames:
                continue
            kpts = np.stack([blk[c]['pose_2d'] for c in sel], axis=1)        # (T, V, 133, 2)
            pose3 = blk[sel[0]]['camera_3d']                                  # (T, 133, 3) in cam0 frame
            kpts17 = coco17_to_h36m17(kpts)[:, :, :17, :]                     # (T, V, 17, 2)
            pose3_17 = coco17_to_h36m17(pose3)[:, :17, :]                     # (T, 17, 3)
            clips_2d.append(kpts17.astype(np.float32))
            clips_3d.append(pose3_17.astype(np.float32))
    print(f'[h3wb] clips = {len(clips_2d)}, total frames = {sum(c.shape[0] for c in clips_2d)}')
    return clips_2d, clips_3d


def normalize(kpts2d, pose3d):
    """Per-clip normalization: 2D centered/scaled by image-ish range; 3D root-relative + mm->m."""
    # 2D: shift to clip centroid, scale by max abs
    c = kpts2d.reshape(-1, 2).mean(axis=0)
    s = max(np.abs(kpts2d - c).max(), 1.0)
    kpts2d = (kpts2d - c) / s
    # 3D: subtract pelvis (joint 0), convert mm -> m
    pose3d = (pose3d - pose3d[:, :1, :]) / 1000.0
    return kpts2d.astype(np.float32), pose3d.astype(np.float32)


def make_windows(arr, frames):
    pad = (frames - 1) // 2
    padded = np.pad(arr, [(pad, pad)] + [(0, 0)] * (arr.ndim - 1), mode='edge')
    return np.stack([padded[i:i + frames] for i in range(arr.shape[0])], axis=0)


def synth_features(kpts2d_window, dim=256, seed=0):
    """Deterministic 256-D image-feature surrogate from 2D keypoints (no raw images)."""
    rng = np.random.default_rng(seed)
    N, F, V, J, _ = kpts2d_window.shape
    proj = rng.standard_normal((J * 2, dim)).astype(np.float32) * 0.05
    flat = kpts2d_window.reshape(N, F, V, J * 2)
    return np.tanh(flat @ proj).astype(np.float32)


def build_dataset(clips_2d, clips_3d, frames, max_frames):
    xs, ys = [], []
    total = 0
    for k2, p3 in zip(clips_2d, clips_3d):
        k2n, p3n = normalize(k2, p3)
        xs.append(make_windows(k2n, frames))
        ys.append(make_windows(p3n, frames))
        total += k2n.shape[0]
        if total >= max_frames:
            break
    x = np.concatenate(xs, axis=0)[:max_frames]
    y = np.concatenate(ys, axis=0)[:max_frames]
    f = synth_features(x)
    return x, f, y


def p_mpjpe(pred, gt):
    """Procrustes-aligned MPJPE (a.k.a. PA-MPJPE / P-MPJPE), in same units as inputs.

    pred, gt: (N, J, 3) numpy arrays.
    """
    mu_p = pred.mean(axis=1, keepdims=True)
    mu_g = gt.mean(axis=1, keepdims=True)
    p0, g0 = pred - mu_p, gt - mu_g
    norm_p = np.linalg.norm(p0, axis=(1, 2), keepdims=True) + 1e-9
    norm_g = np.linalg.norm(g0, axis=(1, 2), keepdims=True) + 1e-9
    p0n, g0n = p0 / norm_p, g0 / norm_g
    H = np.einsum('nji,njk->nik', p0n, g0n)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum('nij,njk->nik', Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    D = np.broadcast_to(np.eye(3), U.shape).copy()
    D[:, -1, -1] = d
    R = np.einsum('nij,njk,nkl->nil', Vt.transpose(0, 2, 1), D, U.transpose(0, 2, 1))
    s = (S * D[:, [0, 1, 2], [0, 1, 2]]).sum(axis=1, keepdims=True)[..., None] * (norm_g / norm_p)
    aligned = s * np.einsum('nij,nkj->nki', R, p0) + mu_g
    return float(np.linalg.norm(aligned - gt, axis=-1).mean())


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    device = pick_device()
    F, V, J = opt.frames, opt.num_view, 17
    view_idx = [int(s) for s in opt.view_indices.split(',')]
    print(f'[mini-h3wb] device={device} frames={F} num_view={V} view_indices={view_idx}')

    clips_2d, clips_3d = load_h3wb(DATA_FILE, view_idx, F)

    # Spec from kế hoạch: ~500-1000 mẫu mini-train
    n_train, n_val = 800, 200
    rng_idx = np.random.default_rng(0)
    perm = rng_idx.permutation(len(clips_2d))
    cut = max(1, int(len(clips_2d) * 0.85))
    train_clips_2d = [clips_2d[i] for i in perm[:cut]]
    train_clips_3d = [clips_3d[i] for i in perm[:cut]]
    val_clips_2d = [clips_2d[i] for i in perm[cut:]]
    val_clips_3d = [clips_3d[i] for i in perm[cut:]]

    x_tr, f_tr, y_tr = build_dataset(train_clips_2d, train_clips_3d, F, n_train)
    x_va, f_va, y_va = build_dataset(val_clips_2d, val_clips_3d, F, n_val)
    print(f'[mini-h3wb] train {x_tr.shape} | val {x_va.shape}')

    model = Model(
        num_frame=F, num_joints=J, in_chans=2,
        embed_dim_ratio=opt.embed_dim_ratio,
        img_embed_dim_ratio=opt.img_embed_dim_ratio,
        depth=opt.depth, num_heads=8, mlp_ratio=2.,
        qkv_bias=True, qk_scale=None, drop_path_rate=0.,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'[mini-h3wb] params={n_params:.1f}M')

    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_fn = nn.MSELoss()
    bs = max(1, min(opt.batch_size, 8))
    epochs = opt.nepoch
    history = {'train': [], 'val': [], 'mpjpe_mm': [], 'p_mpjpe_mm': []}
    best_val = float('inf')
    ckpt = os.path.join(OUT_DIR, 'mini_best.pth')

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(x_tr.shape[0])
        tr = 0.0; nb = 0
        for i in range(0, x_tr.shape[0], bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(x_tr[idx]).to(device)
            fb = torch.from_numpy(f_tr[idx]).to(device)
            yb = torch.from_numpy(y_tr[idx]).to(device)
            pred = model(xb, fb)
            loss = loss_fn(pred, yb)
            optim.zero_grad(); loss.backward(); optim.step()
            tr += loss.item(); nb += 1
        tr /= max(nb, 1)

        model.eval()
        with torch.no_grad():
            va = 0.0; nb = 0; preds_all, gts_all = [], []
            for i in range(0, x_va.shape[0], bs):
                xb = torch.from_numpy(x_va[i:i + bs]).to(device)
                fb = torch.from_numpy(f_va[i:i + bs]).to(device)
                yb = torch.from_numpy(y_va[i:i + bs]).to(device)
                pred = model(xb, fb)
                va += loss_fn(pred, yb).item(); nb += 1
                preds_all.append(pred[:, F // 2].cpu().numpy())
                gts_all.append(yb[:, F // 2].cpu().numpy())
            va /= max(nb, 1)
            P = np.concatenate(preds_all, axis=0)   # (N, J, 3) meters
            G = np.concatenate(gts_all, axis=0)
            mpjpe = float(np.linalg.norm(P - G, axis=-1).mean()) * 1000.0
            pmpjpe = p_mpjpe(P, G) * 1000.0
        history['train'].append(tr); history['val'].append(va)
        history['mpjpe_mm'].append(mpjpe); history['p_mpjpe_mm'].append(pmpjpe)
        print(f'[mini-h3wb] ep {ep+1:02d}/{epochs}  train={tr:.5f}  val={va:.5f}  '
              f'MPJPE={mpjpe:.1f}mm  P-MPJPE={pmpjpe:.1f}mm')
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), ckpt)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history['train'], 'o-', label='train')
    ax[0].plot(history['val'], 's-', label='val')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('MSE (m^2)'); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_title(f'DSVTformer mini-train on H3WB (V={V}, depth={opt.depth})')
    ax[1].plot(history['mpjpe_mm'], 'd-', color='tab:red', label='MPJPE')
    ax[1].plot(history['p_mpjpe_mm'], 's-', color='tab:purple', label='P-MPJPE')
    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('mm'); ax[1].grid(alpha=0.3); ax[1].legend()
    ax[1].set_title('Validation MPJPE / P-MPJPE (root-relative)')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'loss_curve.png')
    plt.savefig(out_png, dpi=120); plt.close()
    print(f'[mini-h3wb] loss curve -> {out_png}')
    print(f'[mini-h3wb] best val = {best_val:.5f}  ckpt -> {ckpt}')

    # Save first val clip as inference test sequence
    test_x = x_va[0:1, opt.frames // 2]   # (1, V, J, 2) center frames? No — keep whole window:
    # Use sequence of 64 center frames from validation windows (they are per-frame windows)
    T_take = min(64, x_va.shape[0])
    np.save(os.path.join(OUT_DIR, 'kpts_2d.npy'), x_va[:T_take, opt.frames // 2].astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'img_feat.npy'), f_va[:T_take, opt.frames // 2].astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'gt_3d.npy'), y_va[:T_take, opt.frames // 2].astype(np.float32))
    print(f'[mini-h3wb] test seq -> outputs_h3wb/{{kpts_2d,img_feat,gt_3d}}.npy ({T_take} frames)')


if __name__ == '__main__':
    main()
