from pathlib import Path
import csv
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

LOW_DIR = "./data/Test/LOW"
HIGH_DIR = "./data/Test/HIGH"

MODEL_PATH = "./models/best/model_best.pth"

OUTPUT_DIR = "./outputs/evaluation"

HEIGHT = 200
WIDTH = 200
DEPTH = 512
OFFSET = 48

PROJECTION = "p99"          # "max", "p99", "mean"
NORMALIZE = "percentile"   # "minmax", "percentile"

BATCH_SIZE = 4
BASE_CHANNELS = 32
NUM_WORKERS = 0
SEED = 42
SSIM_WINDOW_SIZE = 11

NUM_VISUALS = 16
SAVE_PREDICTIONS = True
USE_AMP = True


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_per_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute PSNR for each image.

    Input:
        pred, target: [B, C, H, W], range [0, 1]

    Return:
        [B]
    """
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))

    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim_per_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute local-window SSIM for each image.

    Same formulation as the training code.

    Input:
        pred, target: [B, C, H, W], range [0, 1]

    Return:
        [B]
    """
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    padding = window_size // 2

    mu_x = F.avg_pool2d(
        pred,
        kernel_size=window_size,
        stride=1,
        padding=padding,
    )

    mu_y = F.avg_pool2d(
        target,
        kernel_size=window_size,
        stride=1,
        padding=padding,
    )

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.avg_pool2d(
            pred * pred,
            kernel_size=window_size,
            stride=1,
            padding=padding,
        )
        - mu_x_sq
    )

    sigma_y_sq = (
        F.avg_pool2d(
            target * target,
            kernel_size=window_size,
            stride=1,
            padding=padding,
        )
        - mu_y_sq
    )

    sigma_xy = (
        F.avg_pool2d(
            pred * target,
            kernel_size=window_size,
            stride=1,
            padding=padding,
        )
        - mu_xy
    )

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)

    denominator = (
        (mu_x_sq + mu_y_sq + c1)
        * (sigma_x_sq + sigma_y_sq + c2)
    )

    ssim_map = numerator / (denominator + eps)

    return ssim_map.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)


# ============================================================
# BIN loading functions
# ============================================================

def read_bin_volume(
    path: Path,
    shape: Tuple[int, int, int] = (200, 200, 512),
    dtype=np.uint16,
    offset: int = 48,
) -> np.ndarray:
    """
    Read one raw .bin volume.

    Expected stored order:
        [H, W, D]

    Return:
        volume: float32 numpy array, shape [H, W, D]
    """
    expected_elements = int(np.prod(shape))
    expected_size = (
        expected_elements * np.dtype(dtype).itemsize + offset
    )

    file_size = path.stat().st_size

    if file_size != expected_size:
        raise ValueError(
            f"File size mismatch: {path}\n"
            f"Expected: {expected_size} bytes\n"
            f"Actual  : {file_size} bytes\n"
            f"Check height, width, depth, dtype, or offset."
        )

    raw = np.fromfile(
        path,
        dtype=dtype,
        offset=offset,
    )

    if raw.size != expected_elements:
        raise ValueError(
            f"Element count mismatch: {path}\n"
            f"Expected: {expected_elements}\n"
            f"Actual  : {raw.size}"
        )

    volume = raw.reshape(shape)

    return volume.astype(np.float32)


def make_projection(
    volume: np.ndarray,
    projection: str = "p99",
) -> np.ndarray:
    """
    Convert [H, W, D] volume to [H, W] MAP image.

    projection:
        max  : maximum intensity projection
        p99  : 99th percentile projection
        mean : mean projection
    """
    if volume.ndim != 3:
        raise ValueError(
            f"Expected volume shape [H, W, D], got {volume.shape}"
        )

    if projection == "max":
        img = np.max(volume, axis=2)

    elif projection == "p99":
        img = np.percentile(volume, 99, axis=2)

    elif projection == "mean":
        img = np.mean(volume, axis=2)

    else:
        raise ValueError(
            f"Unsupported projection: {projection}"
        )

    return img.astype(np.float32)


def normalize_image(
    img: np.ndarray,
    method: str = "percentile",
) -> np.ndarray:
    """
    Normalize 2D image to [0, 1].

    method:
        minmax
        percentile
    """
    img = img.astype(np.float32)

    if method == "minmax":
        v_min = float(img.min())
        v_max = float(img.max())

    elif method == "percentile":
        v_min, v_max = np.percentile(
            img,
            [1, 99],
        )

        v_min = float(v_min)
        v_max = float(v_max)

        img = np.clip(
            img,
            v_min,
            v_max,
        )

    else:
        raise ValueError(
            f"Unsupported normalize method: {method}"
        )

    if v_max > v_min:
        img = (img - v_min) / (v_max - v_min)
    else:
        img = img * 0.0

    return img.astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class PairedBinMAPDataset(Dataset):
    """
    LOW .bin volume  -> input MAP image
    HIGH .bin volume -> target MAP image

    Output:
        low  : [1, H, W], float32, [0, 1]
        high : [1, H, W], float32, [0, 1]
        name : paired sample name
    """

    def __init__(
        self,
        low_dir: str,
        high_dir: str,
        shape: Tuple[int, int, int] = (200, 200, 512),
        dtype=np.uint16,
        offset: int = 48,
        projection: str = "p99",
        normalize: str = "percentile",
    ):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)

        self.shape = shape
        self.dtype = dtype
        self.offset = offset

        self.projection = projection
        self.normalize = normalize

        low_files = sorted(
            self.low_dir.glob("*.bin")
        )

        high_files = sorted(
            self.high_dir.glob("*.bin")
        )

        self.pairs = self._make_pairs(
            low_files,
            high_files,
        )

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No paired .bin files found.\n"
                f"LOW dir : {self.low_dir}\n"
                f"HIGH dir: {self.high_dir}\n"
                f"Expected examples:\n"
                f"  Test_000_LOW.bin\n"
                f"  Test_000_HIGH.bin"
            )

        print(
            f"Found {len(self.pairs)} LOW/HIGH bin pairs."
        )

    @staticmethod
    def _normalize_stem(
        path: Path,
    ) -> str:
        """
        Train_000_LOW.bin  -> Train_000
        Train_000_HIGH.bin -> Train_000
        Test_000_LOW.bin   -> Test_000
        Test_000_HIGH.bin  -> Test_000
        """
        stem = path.stem

        for suffix in [
            "_LOW",
            "_HIGH",
            "_low",
            "_high",
        ]:
            stem = stem.replace(
                suffix,
                "",
            )

        return stem

    def _make_pairs(
        self,
        low_files: List[Path],
        high_files: List[Path],
    ) -> List[Tuple[Path, Path]]:

        high_map = {
            self._normalize_stem(p): p
            for p in high_files
        }

        pairs = []

        for low_path in low_files:
            key = self._normalize_stem(
                low_path
            )

            high_path = high_map.get(key)

            if high_path is not None:
                pairs.append(
                    (
                        low_path,
                        high_path,
                    )
                )

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self,
        idx: int,
    ):
        low_path, high_path = self.pairs[idx]

        low_volume = read_bin_volume(
            low_path,
            shape=self.shape,
            dtype=self.dtype,
            offset=self.offset,
        )

        high_volume = read_bin_volume(
            high_path,
            shape=self.shape,
            dtype=self.dtype,
            offset=self.offset,
        )

        low_map = make_projection(
            low_volume,
            projection=self.projection,
        )

        high_map = make_projection(
            high_volume,
            projection=self.projection,
        )

        low_map = normalize_image(
            low_map,
            method=self.normalize,
        )

        high_map = normalize_image(
            high_map,
            method=self.normalize,
        )

        low = torch.from_numpy(
            low_map
        ).unsqueeze(0)

        high = torch.from_numpy(
            high_map
        ).unsqueeze(0)

        name = self._normalize_stem(
            low_path
        )

        return low, high, name


# ============================================================
# Model
# ============================================================

class UNetConv2(nn.Module):
    """
    Conv -> BN -> ReLU
    Conv -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class UNetEnhancer(nn.Module):
    """
    Same grayscale 2D U-Net structure
    used in the training code.

    Input:
        [B, 1, H, W]

    Output:
        [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()

        b = base_channels

        self.conv_1 = UNetConv2(
            in_channels,
            b,
        )

        self.conv_2 = UNetConv2(
            b,
            b * 2,
        )

        self.conv_3 = UNetConv2(
            b * 2,
            b * 4,
        )

        self.conv_4 = UNetConv2(
            b * 4,
            b * 8,
        )

        self.mid_conv = UNetConv2(
            b * 8,
            b * 16,
        )

        self.up_1 = nn.ConvTranspose2d(
            b * 16,
            b * 8,
            kernel_size=2,
            stride=2,
        )

        self.up_2 = nn.ConvTranspose2d(
            b * 8,
            b * 4,
            kernel_size=2,
            stride=2,
        )

        self.up_3 = nn.ConvTranspose2d(
            b * 4,
            b * 2,
            kernel_size=2,
            stride=2,
        )

        self.up_4 = nn.ConvTranspose2d(
            b * 2,
            b,
            kernel_size=2,
            stride=2,
        )

        self.conv_5 = UNetConv2(
            b * 16,
            b * 8,
        )

        self.conv_6 = UNetConv2(
            b * 8,
            b * 4,
        )

        self.conv_7 = UNetConv2(
            b * 4,
            b * 2,
        )

        self.conv_8 = UNetConv2(
            b * 2,
            b,
        )

        self.down = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.end = nn.Conv2d(
            b,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    @staticmethod
    def center_crop_like(
        src: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        _, _, h, w = src.shape
        _, _, th, tw = target.shape

        top = max(
            (h - th) // 2,
            0,
        )

        left = max(
            (w - tw) // 2,
            0,
        )

        return src[
            :,
            :,
            top: top + th,
            left: left + tw,
        ]

    @staticmethod
    def pad_to_even(
        x: torch.Tensor,
    ) -> torch.Tensor:

        pad_h = x.size(2) % 2
        pad_w = x.size(3) % 2

        if pad_h != 0 or pad_w != 0:
            x = F.pad(
                x,
                (
                    0,
                    pad_w,
                    0,
                    pad_h,
                ),
                mode="reflect",
            )

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        padded_x = F.pad(
            x,
            (
                92,
                92,
                92,
                92,
            ),
            mode="reflect",
        )

        conv_1 = self.conv_1(
            padded_x
        )

        pool1 = self.down(
            self.pad_to_even(
                conv_1
            )
        )

        conv_2 = self.conv_2(
            pool1
        )

        pool2 = self.down(
            self.pad_to_even(
                conv_2
            )
        )

        conv_3 = self.conv_3(
            pool2
        )

        pool3 = self.down(
            self.pad_to_even(
                conv_3
            )
        )

        conv_4 = self.conv_4(
            pool3
        )

        pool4 = self.down(
            self.pad_to_even(
                conv_4
            )
        )

        mid = self.mid_conv(
            pool4
        )

        up_1 = self.up_1(
            mid
        )

        up_1 = torch.cat(
            [
                up_1,
                self.center_crop_like(
                    conv_4,
                    up_1,
                ),
            ],
            dim=1,
        )

        conv_5 = self.conv_5(
            up_1
        )

        up_2 = self.up_2(
            conv_5
        )

        up_2 = torch.cat(
            [
                up_2,
                self.center_crop_like(
                    conv_3,
                    up_2,
                ),
            ],
            dim=1,
        )

        conv_6 = self.conv_6(
            up_2
        )

        up_3 = self.up_3(
            conv_6
        )

        up_3 = torch.cat(
            [
                up_3,
                self.center_crop_like(
                    conv_2,
                    up_3,
                ),
            ],
            dim=1,
        )

        conv_7 = self.conv_7(
            up_3
        )

        up_4 = self.up_4(
            conv_7
        )

        up_4 = torch.cat(
            [
                up_4,
                self.center_crop_like(
                    conv_1,
                    up_4,
                ),
            ],
            dim=1,
        )

        conv_8 = self.conv_8(
            up_4
        )

        out = self.end(
            conv_8
        )

        out = self.center_crop_like(
            out,
            x,
        )

        return torch.sigmoid(out)


# ============================================================
# Model loading
# ============================================================

def load_trained_model(
    model_path: str,
    device: torch.device,
) -> UNetEnhancer:
    """
    Load already-trained model_best.pth.

    Supports:
        1. torch.save(model.state_dict(), path)
        2. checkpoint dict with "model_state_dict"
    """

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    model = UNetEnhancer(
        in_channels=1,
        out_channels=1,
        base_channels=BASE_CHANNELS,
    ).to(device)

    saved_data = torch.load(
        path,
        map_location=device,
    )

    if (
        isinstance(saved_data, dict)
        and "model_state_dict" in saved_data
    ):
        model.load_state_dict(
            saved_data["model_state_dict"]
        )

        print(
            "Loaded checkpoint-style model."
        )

        if "epoch" in saved_data:
            print(
                f"Checkpoint epoch: "
                f"{int(saved_data['epoch']) + 1}"
            )

    else:
        model.load_state_dict(
            saved_data
        )

        print(
            "Loaded model state_dict."
        )

    model.eval()

    return model


# ============================================================
# Visualization
# ============================================================

def save_prediction_image(
    pred: torch.Tensor,
    path: Path,
) -> None:
    """
    Save prediction as hot colormap image.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_np = (
        pred.detach()
        .cpu()
        .squeeze()
        .clamp(0, 1)
        .numpy()
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        pred_np,
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
    )

    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_comparison_image(
    low: torch.Tensor,
    pred: torch.Tensor,
    high: torch.Tensor,
    path: Path,
    title: str = "",
) -> None:
    """
    Save:
        LOW | Prediction | HIGH | Absolute Error
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    low_np = (
        low.detach()
        .cpu()
        .squeeze()
        .clamp(0, 1)
        .numpy()
    )

    pred_np = (
        pred.detach()
        .cpu()
        .squeeze()
        .clamp(0, 1)
        .numpy()
    )

    high_np = (
        high.detach()
        .cpu()
        .squeeze()
        .clamp(0, 1)
        .numpy()
    )

    error_np = np.abs(
        pred_np - high_np
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    image_info = [
        (
            low_np,
            "LOW Input",
            0.0,
            1.0,
        ),
        (
            pred_np,
            "Prediction",
            0.0,
            1.0,
        ),
        (
            high_np,
            "HIGH Target",
            0.0,
            1.0,
        ),
        (
            error_np,
            "Absolute Error",
            0.0,
            max(
                float(error_np.max()),
                1e-8,
            ),
        ),
    ]

    for ax, (
        img,
        subtitle,
        vmin,
        vmax,
    ) in zip(
        axes,
        image_info,
    ):
        im = ax.imshow(
            img,
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(
            subtitle
        )

        ax.axis("off")

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

    if title:
        fig.suptitle(
            title
        )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate() -> None:
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = PairedBinMAPDataset(
        low_dir=LOW_DIR,
        high_dir=HIGH_DIR,
        shape=(
            HEIGHT,
            WIDTH,
            DEPTH,
        ),
        dtype=np.uint16,
        offset=OFFSET,
        projection=PROJECTION,
        normalize=NORMALIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # --------------------------------------------------------
    # Load already-trained model
    # --------------------------------------------------------

    model = load_trained_model(
        model_path=MODEL_PATH,
        device=device,
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_dir = Path(
        OUTPUT_DIR
    )

    visual_dir = (
        output_dir
        / "visuals"
    )

    prediction_dir = (
        output_dir
        / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visual_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_PREDICTIONS:
        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        f"Model path        : {MODEL_PATH}"
    )

    print(
        f"Evaluation samples: {len(dataset)}"
    )

    print(
        f"Projection        : {PROJECTION}"
    )

    print(
        f"Normalization     : {NORMALIZE}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_l1 = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    result_rows: List[
        Dict[str, Any]
    ] = []

    saved_visuals = 0

    amp_enabled = (
        USE_AMP
        and device.type == "cuda"
    )

    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc="Evaluation",
    )

    # --------------------------------------------------------
    # Inference + evaluation
    # --------------------------------------------------------

    for (
        low,
        high,
        names,
    ) in progress_bar:

        low = low.to(
            device,
            non_blocking=True,
        )

        high = high.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            pred = model(
                low
            )

        pred = pred.float().clamp(
            0.0,
            1.0,
        )

        high = high.float().clamp(
            0.0,
            1.0,
        )

        l1_each = torch.mean(
            torch.abs(
                pred - high
            ),
            dim=(
                1,
                2,
                3,
            ),
        )

        mse_each = torch.mean(
            (
                pred - high
            ) ** 2,
            dim=(
                1,
                2,
                3,
            ),
        )

        psnr_each = psnr_per_image(
            pred,
            high,
        )

        ssim_each = ssim_per_image(
            pred,
            high,
            window_size=SSIM_WINDOW_SIZE,
            data_range=1.0,
        )

        current_batch_size = low.size(
            0
        )

        total_l1 += (
            l1_each.sum().item()
        )

        total_mse += (
            mse_each.sum().item()
        )

        total_psnr += (
            psnr_each.sum().item()
        )

        total_ssim += (
            ssim_each.sum().item()
        )

        total_samples += (
            current_batch_size
        )

        progress_bar.set_postfix({
            "L1": (
                f"{l1_each.mean().item():.4f}"
            ),
            "PSNR": (
                f"{psnr_each.mean().item():.2f}"
            ),
            "SSIM": (
                f"{ssim_each.mean().item():.4f}"
            ),
        })

        # ----------------------------------------------------
        # Per-image results
        # ----------------------------------------------------

        for i in range(
            current_batch_size
        ):
            name = names[i]

            result_rows.append({
                "filename": name,
                "l1_loss": float(
                    l1_each[i].item()
                ),
                "mse_loss": float(
                    mse_each[i].item()
                ),
                "psnr": float(
                    psnr_each[i].item()
                ),
                "ssim": float(
                    ssim_each[i].item()
                ),
            })

            # Save prediction
            if SAVE_PREDICTIONS:
                save_prediction_image(
                    pred[i],
                    prediction_dir
                    / f"{name}_prediction.png",
                )

            # Save comparison visualization
            if saved_visuals < NUM_VISUALS:
                save_comparison_image(
                    low[i],
                    pred[i],
                    high[i],
                    visual_dir
                    / f"{name}_comparison.png",
                    title=name,
                )

                saved_visuals += 1

    # --------------------------------------------------------
    # Average metrics
    # --------------------------------------------------------

    avg_l1 = (
        total_l1
        / max(
            total_samples,
            1,
        )
    )

    avg_mse = (
        total_mse
        / max(
            total_samples,
            1,
        )
    )

    avg_psnr = (
        total_psnr
        / max(
            total_samples,
            1,
        )
    )

    avg_ssim = (
        total_ssim
        / max(
            total_samples,
            1,
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = (
        output_dir
        / "evaluation_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "l1_loss",
                "mse_loss",
                "psnr",
                "ssim",
            ],
        )

        writer.writeheader()

        writer.writerows(
            result_rows
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "evaluation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Evaluation Results\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            f"Model path   : "
            f"{Path(MODEL_PATH).resolve()}\n"
        )

        f.write(
            f"Samples      : "
            f"{total_samples}\n"
        )

        f.write(
            f"Projection   : "
            f"{PROJECTION}\n"
        )

        f.write(
            f"Normalization: "
            f"{NORMALIZE}\n"
        )

        f.write(
            f"L1 Loss      : "
            f"{avg_l1:.6f}\n"
        )

        f.write(
            f"MSE Loss     : "
            f"{avg_mse:.6f}\n"
        )

        f.write(
            f"PSNR         : "
            f"{avg_psnr:.4f} dB\n"
        )

        f.write(
            f"SSIM         : "
            f"{avg_ssim:.6f}\n"
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\nEvaluation Results"
    )

    print(
        "------------------"
    )

    print(
        f"Model path   : "
        f"{Path(MODEL_PATH).resolve()}"
    )

    print(
        f"Samples      : "
        f"{total_samples}"
    )

    print(
        f"L1 Loss      : "
        f"{avg_l1:.6f}"
    )

    print(
        f"MSE Loss     : "
        f"{avg_mse:.6f}"
    )

    print(
        f"PSNR         : "
        f"{avg_psnr:.4f} dB"
    )

    print(
        f"SSIM         : "
        f"{avg_ssim:.6f}"
    )

    print(
        f"CSV          : "
        f"{csv_path.resolve()}"
    )

    print(
        f"Summary      : "
        f"{summary_path.resolve()}"
    )

    print(
        f"Visuals      : "
        f"{visual_dir.resolve()}"
    )

    if SAVE_PREDICTIONS:
        print(
            f"Predictions  : "
            f"{prediction_dir.resolve()}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate()
