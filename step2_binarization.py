import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path


def safe_resize_by_width(img, target_width=900):
    h, w = img.shape[:2]
    if w == 0:
        return img
    new_h = int(h * target_width / w)
    return cv2.resize(img, (target_width, new_h))


def ensure_black_text_white_bg(bin_img):
    """
    Ensure final binary image is black text on white background.
    Heuristic:
    - if white pixels are too few, invert image
    """
    white_ratio = np.sum(bin_img == 255) / bin_img.size
    if white_ratio < 0.5:
        bin_img = 255 - bin_img
    return bin_img


def step2_binarization():
    
    print(" STEP 2 : BINARIZATION ")
    

    base_dir = Path.cwd()
    step1_dir = base_dir / "STEP1_OUTPUTS"
    manifest_path = step1_dir / "step1_manifest.csv"

    if not manifest_path.exists():
        print(f"ERROR: Step-1 manifest not found: {manifest_path}")
        print("Run: python step1_data_collection.py")
        return

    df = pd.read_csv(manifest_path)

    if df.empty:
        print("ERROR: Step-1 manifest is empty.")
        return

    required_cols = ["filename", "full_path", "transcription"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in step1 manifest: {missing_cols}")
        return

    # Output folders
    out_dir = base_dir / "STEP2_OUTPUTS"
    bin_dir = out_dir / "binarized"
    preview_dir = out_dir / "preview"

    out_dir.mkdir(exist_ok=True)
    bin_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    # Adaptive threshold parameters
    adaptive_block = 31   # must be odd
    adaptive_C = 10

    results = []
    failed_files = []

    for _, row in df.iterrows():
        img_path = str(row["full_path"])
        fname = str(row["filename"])
        transcription = str(row["transcription"]) if "transcription" in row else ""

        line_id = ""
        if "line_id" in df.columns:
            line_id = str(row["line_id"])

        img = cv2.imread(img_path)
        if img is None:
            failed_files.append(fname)
            continue

        if len(img.shape) == 2:
            gray = img.copy()
            channels = 1
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            channels = img.shape[2]

        # Mild smoothing before thresholding
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive thresholding
        bin_img = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block,
            adaptive_C
        )

        # Keep black text on white background
        bin_img = ensure_black_text_white_bg(bin_img)

        # Save binarized image
        out_path = bin_dir / fname
        cv2.imwrite(str(out_path), bin_img)

        h, w = gray.shape[:2]

        results.append({
            "line_id": line_id,
            "filename": fname,
            "input_path": img_path,
            "binarized_path": str(out_path.resolve()),
            "width": w,
            "height": h,
            "channels": channels,
            "transcription": transcription,
            "method": "adaptive_gaussian",
            "adaptive_block": adaptive_block,
            "adaptive_C": adaptive_C
        })

    step2_manifest = pd.DataFrame(results)
    step2_manifest_path = out_dir / "step2_manifest.csv"
    step2_manifest.to_csv(step2_manifest_path, index=False, encoding="utf-8-sig")

    # Preview generation
    for i in range(min(3, len(step2_manifest))):
        try:
            fname = step2_manifest.iloc[i]["filename"]
            orig = cv2.imread(step2_manifest.iloc[i]["input_path"])
            b = cv2.imread(step2_manifest.iloc[i]["binarized_path"], cv2.IMREAD_GRAYSCALE)

            if orig is None or b is None:
                continue

            orig_small = safe_resize_by_width(orig, 900)

            b_color = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
            b_small = safe_resize_by_width(b_color, 900)

            # Add labels on preview
            cv2.putText(orig_small, "Original", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(b_small, "Binarized", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            stacked = np.vstack([orig_small, b_small])
            cv2.imwrite(str(preview_dir / f"preview_{i+1}_{fname}"), stacked)

        except Exception as e:
            print(f"Preview creation failed for sample {i+1}: {e}")

    print(f"Total images in Step-1 manifest : {len(df)}")
    print(f"Total images processed          : {len(step2_manifest)}")
    print(f"Failed to read                  : {len(failed_files)}")

    if failed_files:
        print("\nSample failed files:")
        for f in failed_files[:20]:
            print(" -", f)
        if len(failed_files) > 20:
            print(f" ... and {len(failed_files) - 20} more")

    print("\nOutputs saved:")
    print(f"✔ Binarized images folder : {bin_dir}")
    print(f"✔ Step-2 manifest         : {step2_manifest_path}")
    print(f"✔ Preview images          : {preview_dir}")

    if not step2_manifest.empty:
        print("\nSample Step-2 records:")
        print(step2_manifest[["filename", "transcription", "binarized_path"]].head(5).to_string(index=False))

    


if __name__ == "__main__":
    step2_binarization()