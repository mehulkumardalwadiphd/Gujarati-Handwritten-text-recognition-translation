import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path



def estimate_skew_angle(binary_img):
    # foreground assumed black text on white background
    inv = cv2.bitwise_not(binary_img)
    coords = np.column_stack(np.where(inv > 0))

    if coords.shape[0] < 200:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = float(rect[-1])

    # Normalize to [-45, +45]
    if angle < -45:
        angle = 90 + angle

    # Safety against false 90-degree artifacts
    if abs(angle) > 45:
        angle = 0.0

    # Conservative clamp for handwritten line images
    angle = max(-15.0, min(15.0, angle))

    return angle


def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )
    return rotated


def safe_resize_by_width(img, target_width=900):
    h, w = img.shape[:2]
    if w == 0:
        return img
    new_h = int(h * target_width / w)
    return cv2.resize(img, (target_width, new_h))


def step4_deskew():
    
    print(" STEP 4 : SKEW DETECTION + DESKEW ")
    

    base_dir = Path.cwd()
    step3_dir = base_dir / "STEP3_OUTPUTS"
    manifest_path = step3_dir / "step3_manifest.csv"

    if not manifest_path.exists():
        print(f"ERROR: Step-3 manifest not found: {manifest_path}")
        print("Run: python step3_noise_removal.py")
        return

    df = pd.read_csv(manifest_path)

    if df.empty:
        print("ERROR: Step-3 manifest is empty.")
        return

    required_cols = ["filename", "denoised_path", "transcription"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in step3 manifest: {missing_cols}")
        return

    out_dir = base_dir / "STEP4_OUTPUTS"
    deskew_dir = out_dir / "deskewed"
    preview_dir = out_dir / "preview"

    out_dir.mkdir(exist_ok=True)
    deskew_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    results = []
    failed_files = []
    angles_list = []

    for _, row in df.iterrows():
        fname = str(row["filename"])
        den_path = str(row["denoised_path"])
        transcription = str(row["transcription"]) if "transcription" in row else ""

        line_id = str(row["line_id"]) if "line_id" in df.columns else ""
        input_path = str(row["input_path"]) if "input_path" in df.columns else ""
        binarized_path = str(row["binarized_path"]) if "binarized_path" in df.columns else ""
        width = int(row["width"]) if "width" in df.columns and pd.notna(row["width"]) else None
        height = int(row["height"]) if "height" in df.columns and pd.notna(row["height"]) else None
        channels = int(row["channels"]) if "channels" in df.columns and pd.notna(row["channels"]) else None

        img = cv2.imread(den_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            failed_files.append(fname)
            continue

        angle = estimate_skew_angle(img)

        # Avoid unnecessary interpolation for tiny angles
        if abs(angle) < 0.2:
            deskewed = img.copy()
            applied_angle = 0.0
        else:
            deskewed = rotate_image(img, -angle)
            applied_angle = angle

        out_path = deskew_dir / fname
        cv2.imwrite(str(out_path), deskewed)

        results.append({
            "line_id": line_id,
            "filename": fname,
            "input_path": input_path,
            "binarized_path": binarized_path,
            "denoised_path": str(Path(den_path).resolve()),
            "deskewed_path": str(out_path.resolve()),
            "width": width,
            "height": height,
            "channels": channels,
            "transcription": transcription,
            "skew_angle_deg": round(float(applied_angle), 4)
        })

        angles_list.append(float(applied_angle))

    step4_manifest = pd.DataFrame(results)
    step4_manifest_path = out_dir / "step4_manifest.csv"
    step4_manifest.to_csv(step4_manifest_path, index=False, encoding="utf-8-sig")

    # Save angle statistics
    stats_path = out_dir / "skew_angle_stats.txt"
    if len(angles_list) > 0:
        angles_np = np.array(angles_list, dtype=float)
        stats_text = (
            f"Total processed: {len(angles_np)}\n"
            f"Mean angle (deg): {angles_np.mean():.4f}\n"
            f"Std angle (deg): {angles_np.std():.4f}\n"
            f"Min angle (deg): {angles_np.min():.4f}\n"
            f"Max angle (deg): {angles_np.max():.4f}\n"
        )
    else:
        stats_text = "No angles computed.\n"

    stats_path.write_text(stats_text, encoding="utf-8")

    # Load Step-1 manifest once for preview originals
    step1_manifest_path = base_dir / "STEP1_OUTPUTS" / "step1_manifest.csv"
    step1_df = None
    if step1_manifest_path.exists():
        try:
            step1_df = pd.read_csv(step1_manifest_path)
        except Exception:
            step1_df = None

    # Preview: original + denoised + deskewed
    for i in range(min(3, len(step4_manifest))):
        try:
            fname = step4_manifest.iloc[i]["filename"]
            angle = step4_manifest.iloc[i]["skew_angle_deg"]

            den = cv2.imread(step4_manifest.iloc[i]["denoised_path"], cv2.IMREAD_GRAYSCALE)
            des = cv2.imread(step4_manifest.iloc[i]["deskewed_path"], cv2.IMREAD_GRAYSCALE)

            if den is None or des is None:
                continue

            panels = []

            # Original from Step-1
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

            den_c = cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
            den_c = safe_resize_by_width(den_c, 900)
            cv2.putText(den_c, "Denoised", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            des_c = cv2.cvtColor(des, cv2.COLOR_GRAY2BGR)
            des_c = safe_resize_by_width(des_c, 900)
            cv2.putText(des_c, f"Deskewed | angle={angle:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            panels.append(den_c)
            panels.append(des_c)

            stacked = np.vstack(panels)
            cv2.imwrite(str(preview_dir / f"preview_{i+1}_{fname}"), stacked)

        except Exception as e:
            print(f"Preview creation failed for sample {i+1}: {e}")

    print(f"Total images in Step-3 manifest : {len(df)}")
    print(f"Total images processed          : {len(step4_manifest)}")
    print(f"Failed to read                  : {len(failed_files)}")

    if failed_files:
        print("\nSample failed files:")
        for f in failed_files[:20]:
            print(" -", f)
        if len(failed_files) > 20:
            print(f" ... and {len(failed_files) - 20} more")

    print("\nOutputs saved:")
    print(f"✔ Deskewed images folder : {deskew_dir}")
    print(f"✔ Step-4 manifest        : {step4_manifest_path}")
    print(f"✔ Angle stats            : {stats_path}")
    print(f"✔ Preview images         : {preview_dir}")

    if len(angles_list) > 0:
        print(f"\nSkew angles (deg) range: min={min(angles_list):.2f}, max={max(angles_list):.2f}")

    if not step4_manifest.empty:
        print("\nSample Step-4 records:")
        print(step4_manifest[["filename", "transcription", "skew_angle_deg", "deskewed_path"]].head(5).to_string(index=False))

   


if __name__ == "__main__":
    step4_deskew()