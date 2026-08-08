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


def step3_noise_removal():
    
    print(" STEP 3 : NOISE REMOVAL ")
    

    base_dir = Path.cwd()
    step2_dir = base_dir / "STEP2_OUTPUTS"
    manifest_path = step2_dir / "step2_manifest.csv"

    if not manifest_path.exists():
        print(f"ERROR: Step-2 manifest not found: {manifest_path}")
        print("Run: python step2_binarization.py")
        return

    df = pd.read_csv(manifest_path)

    if df.empty:
        print("ERROR: Step-2 manifest is empty.")
        return

    required_cols = ["filename", "binarized_path", "transcription"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in step2 manifest: {missing_cols}")
        return

    out_dir = base_dir / "STEP3_OUTPUTS"
    den_dir = out_dir / "denoised"
    preview_dir = out_dir / "preview"

    out_dir.mkdir(exist_ok=True)
    den_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    # Median filter as per methodology
    median_ksize = 3  # must be odd

    results = []
    failed_files = []

    for _, row in df.iterrows():
        fname = str(row["filename"])
        bin_path = str(row["binarized_path"])
        transcription = str(row["transcription"]) if "transcription" in row else ""

        line_id = str(row["line_id"]) if "line_id" in df.columns else ""
        input_path = str(row["input_path"]) if "input_path" in df.columns else ""
        width = int(row["width"]) if "width" in df.columns and pd.notna(row["width"]) else None
        height = int(row["height"]) if "height" in df.columns and pd.notna(row["height"]) else None
        channels = int(row["channels"]) if "channels" in df.columns and pd.notna(row["channels"]) else None

        img = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            failed_files.append(fname)
            continue

        den = cv2.medianBlur(img, median_ksize)

        out_path = den_dir / fname
        cv2.imwrite(str(out_path), den)

        results.append({
            "line_id": line_id,
            "filename": fname,
            "input_path": input_path,
            "binarized_path": str(Path(bin_path).resolve()),
            "denoised_path": str(out_path.resolve()),
            "width": width,
            "height": height,
            "channels": channels,
            "transcription": transcription,
            "method": "median_filter",
            "median_ksize": median_ksize
        })

    step3_manifest = pd.DataFrame(results)
    step3_manifest_path = out_dir / "step3_manifest.csv"
    step3_manifest.to_csv(step3_manifest_path, index=False, encoding="utf-8-sig")

    
    step1_manifest_path = base_dir / "STEP1_OUTPUTS" / "step1_manifest.csv"
    step1_df = None
    if step1_manifest_path.exists():
        try:
            step1_df = pd.read_csv(step1_manifest_path)
        except Exception:
            step1_df = None

    # Preview: original vs binarized vs denoised
    for i in range(min(3, len(step3_manifest))):
        try:
            fname = step3_manifest.iloc[i]["filename"]
            bin_img = cv2.imread(step3_manifest.iloc[i]["binarized_path"], cv2.IMREAD_GRAYSCALE)
            den_img = cv2.imread(step3_manifest.iloc[i]["denoised_path"], cv2.IMREAD_GRAYSCALE)

            if bin_img is None or den_img is None:
                continue

            panels = []

            # Original image
            orig_path = None
            if step1_df is not None and "filename" in step1_df.columns and "full_path" in step1_df.columns:
                hit = step1_df[step1_df["filename"] == fname]
                if len(hit) > 0:
                    orig_path = str(hit.iloc[0]["full_path"])

            if orig_path and os.path.exists(orig_path):
                orig = cv2.imread(orig_path)
                if orig is not None:
                    orig_small = safe_resize_by_width(orig, 900)
                    cv2.putText(orig_small, "Original", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                    panels.append(orig_small)

            b_color = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
            b_small = safe_resize_by_width(b_color, 900)
            cv2.putText(b_small, "Binarized", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            d_color = cv2.cvtColor(den_img, cv2.COLOR_GRAY2BGR)
            d_small = safe_resize_by_width(d_color, 900)
            cv2.putText(d_small, "Denoised", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            panels.append(b_small)
            panels.append(d_small)

            stacked = np.vstack(panels)
            cv2.imwrite(str(preview_dir / f"preview_{i+1}_{fname}"), stacked)

        except Exception as e:
            print(f"Preview creation failed for sample {i+1}: {e}")

    print(f"Total images in Step-2 manifest : {len(df)}")
    print(f"Total images processed          : {len(step3_manifest)}")
    print(f"Failed to read                  : {len(failed_files)}")

    if failed_files:
        print("\nSample failed files:")
        for f in failed_files[:20]:
            print(" -", f)
        if len(failed_files) > 20:
            print(f" ... and {len(failed_files) - 20} more")

    print("\nOutputs saved:")
    print(f"✔ Denoised images folder : {den_dir}")
    print(f"✔ Step-3 manifest        : {step3_manifest_path}")
    print(f"✔ Preview images         : {preview_dir}")

    if not step3_manifest.empty:
        print("\nSample Step-3 records:")
        print(step3_manifest[["filename", "transcription", "denoised_path"]].head(5).to_string(index=False))

    


if __name__ == "__main__":
    step3_noise_removal()