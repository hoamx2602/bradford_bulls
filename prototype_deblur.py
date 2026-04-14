"""
Prototype: Deep learning single-image deblurring using NAFNet.

Uses NAFNet architecture directly (no basicsr dependency).

Usage:
    python prototype_deblur.py --video videos/M06_black_1080p.mp4 \
                               --frame 462 835 \
                               --output output/deblur/
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ─── NAFNet Architecture (standalone) ────────────────────────────────

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.reshape(1, C, 1, 1) * y + bias.reshape(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.reshape(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
               grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1,
                               groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0),
        )
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0)
        self.sg2 = SimpleGate()

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. \
            else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. \
            else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=12,
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                              nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups,
                                          encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ─── Utilities ───────────────────────────────────────────────────────

def extract_frame(video_path: str, frame_num: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def compute_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def load_nafnet(device):
    model = NAFNet(
        img_channel=3, width=64, middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
    )

    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / "NAFNet-GoPro-width64.pth"

    if not weights_path.exists():
        print("  Downloading NAFNet pretrained weights...")
        url = ("https://github.com/megvii-research/NAFNet/releases/download/"
               "v0.0.1/NAFNet-GoPro-width64.pth")
        import urllib.request
        urllib.request.urlretrieve(url, str(weights_path))
        print(f"  Saved to {weights_path}")

    checkpoint = torch.load(str(weights_path), map_location="cpu",
                            weights_only=False)
    if "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f"  NAFNet loaded on {device}")
    return model


def deblur_tile(model, tile_float, device):
    """Process a single tile through NAFNet."""
    tensor = torch.from_numpy(tile_float).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
    return output.squeeze(0).permute(1, 2, 0).cpu().numpy()


def deblur_image(model, img_bgr, device, tile_size=256, overlap=32):
    """Run NAFNet on full image using tiling."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Process with tiles
    result = np.zeros_like(img_rgb, dtype=np.float64)
    weight_map = np.zeros((h, w, 1), dtype=np.float64)

    step = tile_size - overlap
    y_positions = list(range(0, max(h - tile_size + 1, 1), step))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    x_positions = list(range(0, max(w - tile_size + 1, 1), step))
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)

    total = len(y_positions) * len(x_positions)
    print(f"  Tiling: {len(y_positions)}x{len(x_positions)} = {total} tiles "
          f"({tile_size}x{tile_size}, overlap={overlap})")

    count = 0
    for y in y_positions:
        for x in x_positions:
            tile = img_rgb[y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[:2]

            out_tile = deblur_tile(model, tile, device)
            out_tile = out_tile[:th, :tw]

            # Feathered blending mask
            mask = np.ones((th, tw, 1), dtype=np.float64)
            feather = min(overlap // 2, 16)
            if feather > 1:
                for f in range(feather):
                    a = f / feather
                    mask[f, :] *= a
                    mask[th - 1 - f, :] *= a
                    mask[:, f] *= a
                    mask[:, tw - 1 - f] *= a

            result[y:y + th, x:x + tw] += out_tile.astype(np.float64) * mask
            weight_map[y:y + th, x:x + tw] += mask

            count += 1
            if count % 10 == 0 or count == total:
                print(f"    {count}/{total} tiles done")

    weight_map = np.maximum(weight_map, 1e-10)
    result = result / weight_map
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def run_deblur(video_path, frame_nums, output_dir="output/deblur"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading NAFNet...")
    model = load_nafnet(device)

    for fn in frame_nums:
        print(f"\n{'='*50}")
        print(f"Processing frame {fn}")
        print(f"{'='*50}")

        frame = extract_frame(video_path, fn)
        if frame is None:
            print(f"  ERROR: Could not read frame {fn}")
            continue

        orig_sharp = compute_sharpness(frame)
        print(f"  Original sharpness: {orig_sharp:.1f}")

        deblurred = deblur_image(model, frame, device)
        deblur_sharp = compute_sharpness(deblurred)
        improvement = deblur_sharp / max(orig_sharp, 1)
        print(f"  Deblurred sharpness: {deblur_sharp:.1f} ({improvement:.1f}x)")

        # Save outputs
        cv2.imwrite(str(output_dir / f"f{fn:06d}_original.jpg"),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(output_dir / f"f{fn:06d}_deblurred.jpg"),
                    deblurred, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Full comparison
        comp_orig = frame.copy()
        comp_deblur = deblurred.copy()
        cv2.putText(comp_orig, "ORIGINAL", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(comp_deblur, "NAFNet DEBLURRED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        comp = np.hstack([comp_orig, comp_deblur])
        cv2.imwrite(str(output_dir / f"f{fn:06d}_COMPARISON.jpg"),
                    comp, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Crop comparisons
        h, w = frame.shape[:2]
        crops = {
            "center": (h // 3, h * 2 // 3, w // 3, w * 2 // 3),
            "upper_left": (0, h // 2, 0, w // 2),
        }
        for name, (y1, y2, x1, x2) in crops.items():
            oc = frame[y1:y2, x1:x2].copy()
            dc = deblurred[y1:y2, x1:x2].copy()
            cv2.putText(oc, "ORIG", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(dc, "DEBLURRED", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            crop_comp = np.hstack([oc, dc])
            cv2.imwrite(str(output_dir / f"f{fn:06d}_crop_{name}.jpg"),
                        crop_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\nAll outputs: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAFNet deblurring")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, nargs="+", required=True)
    parser.add_argument("--output", default="output/deblur")
    args = parser.parse_args()

    run_deblur(args.video, args.frame, args.output)
