import os
import cv2
import pandas as pd
from pathlib import Path


def build_text_mapping(google_text_root):
    text_mapping = {}

    for txt_path in google_text_root.rglob("*.txt"):
        # only process text folders
        if txt_path.parent.name.lower() != "text":
            continue

        line_id = txt_path.stem
        words = []

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    # first token = image name, remaining part = Gujarati text
                    word = " ".join(parts[1:]).strip()
                    if word:
                        words.append(word)

            sentence = " ".join(words).strip()

            if sentence:
                text_mapping[line_id] = sentence

        except Exception as e:
            print(f"Error reading {txt_path}: {e}")

    return text_mapping


def step1_data_collection():
    
    print(" STEP 1 : DATA COLLECTION ")
    

    base_dir = Path.cwd()

    dataset_dir = base_dir / "Dataset"
    google_root = dataset_dir / "1_New_Annoatation_Google" / "01_gujarati_word_images"
    manual_root = dataset_dir / "2_New_Manually_Annotated"

    if not dataset_dir.exists():
        print(f"ERROR: Dataset folder not found at: {dataset_dir}")
        return

    if not google_root.exists():
        print(f"ERROR: Google annotation folder not found at: {google_root}")
        return

    if not manual_root.exists():
        print(f"ERROR: Manual annotated folder not found at: {manual_root}")
        return

    out_dir = base_dir / "STEP1_OUTPUTS"
    out_dir.mkdir(exist_ok=True)

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    print("Building text mapping from annotation files...")
    text_mapping = build_text_mapping(google_root)
    print(f"Total line-text entries found: {len(text_mapping)}")

    line_image_files = []
    for img_path in manual_root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in valid_ext:
            # Prefer line images inside images folder
            if "images" in [p.lower() for p in img_path.parts]:
                line_image_files.append(img_path)

    line_image_files = sorted(line_image_files)

    if len(line_image_files) == 0:
        print(f"ERROR: No line images found inside: {manual_root}")
        return

    records = []
    failed = []
    unmatched_images = []

    for idx, img_path in enumerate(line_image_files, start=1):
        img = cv2.imread(str(img_path))

        if img is None:
            failed.append(img_path.name)
            continue

        line_id = img_path.stem

        if line_id not in text_mapping:
            unmatched_images.append(img_path.name)
            continue

        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) == 3 else 1

        records.append({
            "index": idx,
            "line_id": line_id,
            "filename": img_path.name,
            "full_path": str(img_path.resolve()),
            "width": w,
            "height": h,
            "channels": c,
            "text_type": "handwritten",
            "language": "Gujarati",
            "transcription": text_mapping[line_id]
        })

    manifest_df = pd.DataFrame(records)

    manifest_path = out_dir / "step1_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    gt_df = manifest_df[["filename", "transcription"]].copy()
    gt_df["notes"] = ""

    gt_path = out_dir / "ground_truth.csv"
    gt_df.to_csv(gt_path, index=False, encoding="utf-8-sig")

    print(f"Base directory             : {base_dir}")
    print(f"Dataset directory          : {dataset_dir}")
    print(f"Manual image root          : {manual_root}")
    print(f"Google annotation root     : {google_root}")
    print(f"Total line images found    : {len(line_image_files)}")
    print(f"Total text entries built   : {len(text_mapping)}")
    print(f"Matched usable samples     : {len(manifest_df)}")
    print(f"Failed image reads         : {len(failed)}")
    print(f"Unmatched line images      : {len(unmatched_images)}")

    if failed:
        print("\nNot readable files:")
        for f in failed[:30]:
            print(" -", f)
        if len(failed) > 30:
            print(f" ... and {len(failed) - 30} more")

    if unmatched_images:
        print("\nSample unmatched line images:")
        for f in unmatched_images[:30]:
            print(" -", f)
        if len(unmatched_images) > 30:
            print(f" ... and {len(unmatched_images) - 30} more")

    print("\nFiles created:")
    print("✔", manifest_path)
    print("✔", gt_path)

    print("\nSample records:")
    if not manifest_df.empty:
        print(manifest_df[["filename", "transcription"]].head(10).to_string(index=False))
    else:
        print("No matched records created.")

 


if __name__ == "__main__":
    step1_data_collection()