import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# CONFIG

RANDOM_STATE = 42
AUGMENT_TRAIN = True
AUG_COPIES_PER_IMAGE = 3   # number of augmented copies per train image



# HELPERS

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clip_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def safe_imread_gray(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def safe_resize_keep(img, target_width=900):
    h, w = img.shape[:2]
    if w == 0:
        return img
    new_h = int(h * target_width / w)
    return cv2.resize(img, (target_width, new_h))



# AUGMENTATIONS FOR HANDWRITTEN LINE IMAGES

def aug_brightness_contrast(img):
    alpha = random.uniform(0.90, 1.15)
    beta = random.uniform(-15, 15)
    return clip_uint8(img.astype(np.float32) * alpha + beta)


def aug_rotate(img):
    angle = random.uniform(-2.0, 2.0)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )


def aug_translate(img):
    h, w = img.shape[:2]
    tx = random.randint(-5, 5)
    ty = random.randint(-3, 3)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )


def aug_blur(img):
    if random.random() < 0.5:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img


def aug_noise(img):
    if random.random() < 0.5:
        sigma = random.uniform(2, 6)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        return clip_uint8(img.astype(np.float32) + noise)
    return img


def aug_morph(img):
    if random.random() < 0.5:
        k = random.choice([1, 2])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        if random.random() < 0.5:
            return cv2.erode(img, kernel, iterations=1)
        else:
            return cv2.dilate(img, kernel, iterations=1)
    return img


def augment_line_image(img):
    out = img.copy()
    out = aug_rotate(out)
    out = aug_translate(out)
    out = aug_brightness_contrast(out)
    out = aug_blur(out)
    out = aug_noise(out)
    out = aug_morph(out)
    return out



# STEP 5 MAIN

def step5_prepare_split_augment():
    
    print(" STEP 5 : LINE DATASET PREPARATION + SPLIT + TRAIN AUGMENTATION ")
    

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    base_dir = Path.cwd()
    step4_manifest_path = base_dir / "STEP4_OUTPUTS" / "step4_manifest.csv"

    if not step4_manifest_path.exists():
        print(f"ERROR: Step-4 manifest not found: {step4_manifest_path}")
        print("Run: python step4_deskew.py")
        return

    df = pd.read_csv(step4_manifest_path)

    if df.empty:
        print("ERROR: Step-4 manifest is empty.")
        return

    required_cols = ["filename", "deskewed_path", "transcription"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in step4 manifest: {missing_cols}")
        return

    
    # OUTPUT FOLDERS
    
    out_dir = base_dir / "STEP5_OUTPUTS"
    final_lines_dir = out_dir / "final_lines"
    split_dir = out_dir / "splits"
    aug_dir = out_dir / "train_aug"
    preview_dir = out_dir / "preview"

    ensure_dir(out_dir)
    ensure_dir(final_lines_dir)
    ensure_dir(split_dir)
    ensure_dir(aug_dir)
    ensure_dir(preview_dir)

    
    # PART A: VALIDATE AND BUILD FINAL LINE MANIFEST
    
    records = []
    failed_files = []

    for _, row in df.iterrows():
        fname = str(row["filename"])
        deskewed_path = Path(str(row["deskewed_path"]))
        transcription = str(row["transcription"]).strip()

        if not deskewed_path.exists():
            failed_files.append(fname)
            continue

        img = safe_imread_gray(deskewed_path)
        if img is None:
            failed_files.append(fname)
            continue

        h, w = img.shape[:2]

        line_id = str(row["line_id"]) if "line_id" in df.columns else deskewed_path.stem
        input_path = str(row["input_path"]) if "input_path" in df.columns else ""
        binarized_path = str(row["binarized_path"]) if "binarized_path" in df.columns else ""
        denoised_path = str(row["denoised_path"]) if "denoised_path" in df.columns else ""
        skew_angle_deg = float(row["skew_angle_deg"]) if "skew_angle_deg" in df.columns and pd.notna(row["skew_angle_deg"]) else 0.0

        # copy processed line image into final_lines folder
        out_img_path = final_lines_dir / fname
        cv2.imwrite(str(out_img_path), img)

        records.append({
            "line_id": line_id,
            "filename": fname,
            "input_path": input_path,
            "binarized_path": binarized_path,
            "denoised_path": denoised_path,
            "deskewed_path": str(out_img_path.resolve()),
            "width": w,
            "height": h,
            "transcription": transcription,
            "skew_angle_deg": skew_angle_deg
        })

    final_df = pd.DataFrame(records)
    final_manifest_path = out_dir / "step5_final_lines_manifest.csv"
    final_df.to_csv(final_manifest_path, index=False, encoding="utf-8-sig")

    if final_df.empty:
        print("ERROR: No valid line images available after Step-4.")
        return

    
    # PART B: SPLIT 70 / 15 / 15
    
    train_df, temp_df = train_test_split(
        final_df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_csv = split_dir / "train_lines.csv"
    val_csv = split_dir / "val_lines.csv"
    test_csv = split_dir / "test_lines.csv"

    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")

   
    # PART C: AUGMENT ONLY TRAIN SET
    
    train_aug_rows = []
    preview_count = 0
    aug_failed = 0

    if AUGMENT_TRAIN:
        for _, row in train_df.iterrows():
            src_path = Path(str(row["deskewed_path"]))
            img = safe_imread_gray(src_path)

            if img is None:
                aug_failed += 1
                continue

            fname_stem = src_path.stem
            suffix = src_path.suffix

            # save original copy
            orig_name = f"{fname_stem}_orig{suffix}"
            orig_out_path = aug_dir / orig_name
            cv2.imwrite(str(orig_out_path), img)

            orig_row = row.to_dict()
            orig_row["variant"] = "orig"
            orig_row["augmented"] = 0
            orig_row["line_image_path"] = str(orig_out_path.resolve())
            train_aug_rows.append(orig_row)

            # augmented copies
            for k in range(1, AUG_COPIES_PER_IMAGE + 1):
                aug_img = augment_line_image(img)
                aug_name = f"{fname_stem}_aug{k}{suffix}"
                aug_out_path = aug_dir / aug_name
                cv2.imwrite(str(aug_out_path), aug_img)

                aug_row = row.to_dict()
                aug_row["variant"] = f"aug{k}"
                aug_row["augmented"] = 1
                aug_row["line_image_path"] = str(aug_out_path.resolve())
                train_aug_rows.append(aug_row)

                if preview_count < 12:
                    orig_vis = safe_resize_keep(img, 900)
                    aug_vis = safe_resize_keep(aug_img, 900)

                    orig_vis = cv2.cvtColor(orig_vis, cv2.COLOR_GRAY2BGR)
                    aug_vis = cv2.cvtColor(aug_vis, cv2.COLOR_GRAY2BGR)

                    cv2.putText(orig_vis, "Original", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(aug_vis, f"Augmented {k}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                    preview = np.vstack([orig_vis, aug_vis])
                    cv2.imwrite(str(preview_dir / f"preview_{preview_count+1:02d}.png"), preview)
                    preview_count += 1

        train_aug_df = pd.DataFrame(train_aug_rows)
    else:
        train_aug_df = train_df.copy()
        train_aug_df["variant"] = "orig"
        train_aug_df["augmented"] = 0
        train_aug_df["line_image_path"] = train_aug_df["deskewed_path"]

    train_aug_csv = split_dir / "train_lines_aug.csv"
    train_aug_df.to_csv(train_aug_csv, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------
    print(f"Total Step-4 samples           : {len(df)}")
    print(f"Valid final line samples       : {len(final_df)}")
    print(f"Failed during validation       : {len(failed_files)}")

    print("\nSplit summary:")
    print(f"Train samples                  : {len(train_df)}")
    print(f"Validation samples             : {len(val_df)}")
    print(f"Test samples                   : {len(test_df)}")

    if AUGMENT_TRAIN:
        print(f"\nTrain augmentation enabled     : Yes")
        print(f"Augmented copies per image     : {AUG_COPIES_PER_IMAGE}")
        print(f"Augmentation failed reads      : {aug_failed}")
        print(f"Final train rows with aug      : {len(train_aug_df)}")
        print(f"Expected approx                : {len(train_df) * (AUG_COPIES_PER_IMAGE + 1)}")
    else:
        print(f"\nTrain augmentation enabled     : No")

    print("\nOutputs saved:")
    print(f"✔ Final line images folder     : {final_lines_dir}")
    print(f"✔ Final line manifest          : {final_manifest_path}")
    print(f"✔ Train CSV                    : {train_csv}")
    print(f"✔ Validation CSV               : {val_csv}")
    print(f"✔ Test CSV                     : {test_csv}")
    print(f"✔ Train augmented CSV          : {train_aug_csv}")
    print(f"✔ Train augmented images       : {aug_dir}")
    print(f"✔ Preview images               : {preview_dir}")

    if not final_df.empty:
        print("\nSample final line records:")
        print(final_df[["filename", "transcription", "deskewed_path"]].head(5).to_string(index=False))

  


if __name__ == "__main__":
    step5_prepare_split_augment()