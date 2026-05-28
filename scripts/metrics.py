import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================

NUM_CLASSES = 10

ALPHA = 0.3
LAMBDA_PENALTY = 1.5

BETA_MACULA = 0.5
BETA_WIDEFIELD = 0.5

TAU = 0.02

SEEN_VENDORS = [
    "Maestro2",
    "Spectralis",
    "Cirrus",
]

UNSEEN_VENDORS = [
    "Triton",
]


# ============================================================
# Dice
# ============================================================

def dice_score(pred, gt, num_classes=NUM_CLASSES, eps=1e-6):

    dices = []

    for c in range(num_classes):

        pred_c = (pred == c)
        gt_c = (gt == c)

        intersection = np.logical_and(pred_c, gt_c).sum()

        union = pred_c.sum() + gt_c.sum()

        dice = (2.0 * intersection + eps) / (union + eps)

        dices.append(dice)

    return np.array(dices)


# ============================================================
# Surface distance
# ============================================================

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt
)

def extract_boundary(mask):

    eroded = binary_erosion(mask)

    boundary = mask ^ eroded

    return boundary


def surface_distance(pred_mask, gt_mask):

    pred_boundary = extract_boundary(pred_mask)
    gt_boundary = extract_boundary(gt_mask)

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return np.nan

    # Distance transform
    gt_dist = distance_transform_edt(~gt_boundary)
    pred_dist = distance_transform_edt(~pred_boundary)

    pred_to_gt = gt_dist[pred_boundary]

    gt_to_pred = pred_dist[gt_boundary]

    return (
        pred_to_gt.mean() + gt_to_pred.mean()) / 2.0


# ============================================================
# MASD normalized by image height
# ============================================================

def masd_per_class(pred, gt):

    H = pred.shape[0]

    masd = []

    for c in range(NUM_CLASSES):

        pred_c = (pred == c)
        gt_c = (gt == c)

        d = surface_distance(pred_c, gt_c)

        if np.isnan(d):
            masd.append(np.nan)

        else:
            masd.append(d / H)

    return np.array(masd)


# ============================================================
# MASD score
# ============================================================

def masd_to_score(masd):

    return np.exp(-masd / TAU)


# ============================================================
# Image score
# ============================================================

def compute_image_score(pred, gt):

    dice = dice_score(pred, gt)

    masd = masd_per_class(pred, gt)

    masd_score = masd_to_score(masd)

    masd_score = np.nan_to_num(masd_score, nan=0.0)

    layer_score = 0.5 * (dice + masd_score)

    image_score = layer_score.mean()

    return image_score


# ============================================================
# Anatomy normalization
# ============================================================

def normalize_anatomy(anatomy):

    anatomy = str(anatomy).lower()

    if "wide" in anatomy:
        return "WideField"

    elif "macula" in anatomy:
        return "Macula"

    else:
        raise ValueError(f"Unknown anatomy: {anatomy}")



def visualize_example(gt, pred, image_score, save_path, title="Example"):

    dice = dice_score(pred, gt)
    masd = masd_per_class(pred, gt)
    masd_score = masd_to_score(masd)

    # safe formatting
    dice = np.nan_to_num(dice, nan=0.0)
    masd = np.nan_to_num(masd, nan=0.0)
    masd_score = np.nan_to_num(masd_score, nan=0.0)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # --------------------------------------------------------
    # Input proxy (we don't have raw OCT image, so show GT)
    # --------------------------------------------------------
    axes[0, 0].imshow(gt*20, cmap="gray")
    axes[0, 0].set_title("GT (proxy image)")
    axes[0, 0].axis("off")

    # --------------------------------------------------------
    # GT mask
    # --------------------------------------------------------
    axes[0, 1].imshow(gt, cmap="tab10", vmin=0, vmax=9)
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis("off")

    # --------------------------------------------------------
    # Prediction mask
    # --------------------------------------------------------
    axes[0, 2].imshow(pred, cmap="tab10", vmin=0, vmax=9)
    axes[0, 2].set_title("Prediction Mask")
    axes[0, 2].axis("off")
    
    
    # --------------------------------------------------------
    # Dice per class
    # --------------------------------------------------------
    axes[1, 0].bar(np.arange(len(dice)), dice)
    axes[1, 0].set_title("Dice per Class")
    axes[1, 0].set_ylim(0, 1)

    # --------------------------------------------------------
    # MASD per class
    # --------------------------------------------------------
    axes[1, 1].bar(np.arange(len(masd_score)), masd_score)
    axes[1, 1].set_title("MASD per Class")

    # --------------------------------------------------------
    # Summary text panel
    # --------------------------------------------------------
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        0.8,
        f"Image Score: {image_score:.4f}",
        fontsize=12
    )
    axes[1, 2].text(
        0.0,
        0.6,
        f"Mean Dice: {dice.mean():.4f}",
        fontsize=12
    )
    axes[1, 2].text(
        0.0,
        0.4,
        f"Mean MASD: {masd.mean():.4f}",
        fontsize=12
    )
    
    plt.suptitle(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

def main(args):

    gt_dir = Path(args.gt_dir)

    pred_dir = Path(args.pred_dir)

    csv_path = Path(args.csv_path)

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    df = df[df["release_mask_name"].notna()].copy()

    print(f"[INFO] Evaluation samples: {len(df)}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):    

        gt_name = row["release_mask_name"]

        pred_name = gt_name

        gt_path = gt_dir / gt_name

        pred_path = pred_dir / pred_name

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not gt_path.exists():

            print(f"[WARNING] Missing GT: {gt_path}")

            continue

        if not pred_path.exists():

            print(f"[WARNING] Missing prediction: {pred_path}")

            continue

        # ----------------------------------------------------
        # Load GT
        # ----------------------------------------------------

        gt = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE
        )

        assert gt is not None

        # ----------------------------------------------------
        # Load prediction
        # ----------------------------------------------------

        pred = cv2.imread(
            str(pred_path),
            cv2.IMREAD_UNCHANGED
        )

        assert pred is not None

        # ----------------------------------------------------
        # Handle RGB prediction
        # ----------------------------------------------------

        if pred.ndim == 3:

            pred = pred[..., 0]

        # ----------------------------------------------------
        # Resize to GT size
        # ----------------------------------------------------
        if pred.shape != gt.shape:

            print(
                f"[WARNING] Resizing prediction "
                f"{pred_name} from {pred.shape} to {gt.shape}"
            )

            pred = cv2.resize(
                pred,
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        pred = pred.astype(np.uint8)

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        unique_vals = np.unique(pred)

        assert pred.dtype == np.uint8, (
            f"Prediction must be uint8, got {pred.dtype}"
        )

        assert unique_vals.min() >= 0 and unique_vals.max() < NUM_CLASSES, (
            f"Invalid labels in {pred_name}: {unique_vals}"
        )        

        # ----------------------------------------------------
        # Compute score
        # ----------------------------------------------------

        image_score = compute_image_score(pred, gt)

        records.append({
            "release_mask_name": gt_name,
            "device": row["device"],
            "status": row["status"],
            "anatomy": normalize_anatomy(row["group"]),
            "image_score": image_score,
        })



        # ----------------------------------------------------
        # Visualization (only sample 1 healthy + 1 diseased)
        # ----------------------------------------------------

        if not hasattr(main, "saved_vis"):
            main.saved_vis = {"healthy": False, "diseased": False}

        status = row["status"]

        if status in main.saved_vis and not main.saved_vis[status]:

            save_path = f"vis_{status}.png"

            visualize_example(
                gt=gt,
                pred=pred,
                image_score=image_score,
                save_path=save_path,
                title=f"{str(status).upper()} Example"
            )

            main.saved_vis[status] = True


    # --------------------------------------------------------
    # Score DataFrame
    # --------------------------------------------------------

    score_df = pd.DataFrame(records)

    print(f"[INFO] Successfully evaluated: {len(score_df)}")
    score_df.to_csv(
        "evaluation_results.csv",
        index=False
    )
    print("[INFO] Saved evaluation_results.csv")

    # --------------------------------------------------------
    # Cohort aggregation
    # --------------------------------------------------------

    cohort_scores = defaultdict(list)

    for _, row in score_df.iterrows():

        key = (
            row["anatomy"],
            row["device"],
            row["status"],
        )

        cohort_scores[key].append(row["image_score"])

    cohort_scores = {
        k: np.mean(v)
        for k, v in cohort_scores.items()
    }

    # --------------------------------------------------------
    # Vendor scores
    # --------------------------------------------------------

    vendor_scores = {}

    tmp = defaultdict(dict)

    for (anatomy, vendor, status), score in cohort_scores.items():

        tmp[(anatomy, vendor)][status] = score

    for (anatomy, vendor), vals in tmp.items():

        healthy_score = vals.get("healthy", 0.0)

        diseased_score = vals.get("diseased", 0.0)

        combined = (
            ALPHA * healthy_score
            + (1.0 - ALPHA) * diseased_score
        )

        vendor_scores[(anatomy, vendor)] = combined

    # --------------------------------------------------------
    # Anatomy scores
    # --------------------------------------------------------

    anatomy_scores = {}

    for anatomy in ["Macula", "WideField"]:

        anatomy_vendor_scores = {
            vendor: score
            for (a, vendor), score in vendor_scores.items()
            if a == anatomy
        }

        if len(anatomy_vendor_scores) == 0:

            anatomy_scores[anatomy] = 0.0

            continue

        overall = np.mean(
            list(anatomy_vendor_scores.values())
        )

        seen_vals = [
            anatomy_vendor_scores[v]
            for v in SEEN_VENDORS
            if v in anatomy_vendor_scores
        ]

        seen_mean = np.mean(seen_vals)

        penalty = 0.0

        for v in UNSEEN_VENDORS:

            if v not in anatomy_vendor_scores:
                continue

            penalty += max(
                0.0,
                seen_mean - anatomy_vendor_scores[v]
            )

        penalty /= max(len(UNSEEN_VENDORS), 1)

        final_score = max(
            0.0, 
            overall - LAMBDA_PENALTY * penalty
        )

        anatomy_scores[anatomy] = final_score

    # --------------------------------------------------------
    # Final challenge score
    # --------------------------------------------------------

    final_score = (
        BETA_MACULA * anatomy_scores.get("Macula", 0.0)
        + BETA_WIDEFIELD * anatomy_scores.get("WideField", 0.0)
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n===================================================")
    print("FINAL RESULTS")
    print("===================================================\n")

    print("Anatomy Scores:")

    for k, v in anatomy_scores.items():

        print(f"{k}: {v:.4f}")

    print("\nFinal Challenge Score:")
    print(f"{final_score:.4f}")

    print("\n===================================================")


# ============================================================
# Argparse
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Ground truth mask directory"
    )

    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Prediction directory"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Validation release CSV"
    )

    args = parser.parse_args()

    main(args)
