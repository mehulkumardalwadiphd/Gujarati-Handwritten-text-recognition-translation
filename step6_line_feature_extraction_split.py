import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image



# SETTINGS

D_FEATURE = 2048



# DEVICE

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# MODEL

def build_model(device):
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # keep convolutional backbone only
    modules = list(resnet.children())[:-2]
    model = nn.Sequential(*modules)

    model.eval()
    model.to(device)
    return model



# PREPROCESS

def build_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])



# IMAGE READER

def safe_read_gray(path: Path):
    try:
        img = Image.open(path).convert("L")
        img = img.convert("RGB")
        return img
    except Exception:
        return None



# RESOLVE IMAGE PATH COLUMN

def resolve_image_path(row, df_columns):
    """
    For train augmented split:
        uses line_image_path
    For val/test split:
        uses deskewed_path
    """
    if "line_image_path" in df_columns and pd.notna(row.get("line_image_path", None)):
        return str(row["line_image_path"])

    if "deskewed_path" in df_columns and pd.notna(row.get("deskewed_path", None)):
        return str(row["deskewed_path"])

    return None





def resolve_sample_name(row, df_columns):
    filename = str(row["filename"]) if "filename" in df_columns else "sample"
    stem = Path(filename).stem

    variant = "orig"
    if "variant" in df_columns and pd.notna(row["variant"]):
        variant = str(row["variant"])

    return f"{stem}_{variant}"




def feature_map_to_sequence(feat_map: torch.Tensor) -> torch.Tensor:
    """
    Convert spatial feature map into a flattened visual sequence.

    Input:
        feat_map: (1, 2048, H, W)

    Output:
        feat_seq: (H*W, 2048)
    """
    feat_map = feat_map.squeeze(0)            # (2048, H, W)
    c, h, w = feat_map.shape

    feat_seq = feat_map.permute(1, 2, 0)      # (H, W, 2048)
    feat_seq = feat_seq.reshape(h * w, c)     # (H*W, 2048)

    return feat_seq


# FEATURE EXTRACTION FOR ONE SPLIT

def extract_features_for_manifest(manifest_csv: Path, out_dir: Path, split_name: str, device):
    df = pd.read_csv(manifest_csv)

    required_cols = ["filename", "transcription"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{split_name}] Missing required columns {missing} in {manifest_csv.name}. "
            f"Found: {df.columns.tolist()}"
        )

    preprocess = build_preprocess()
    model = build_model(device)

    feat_dir = out_dir / split_name / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = 0
    seq_lens = []

    print(f"\n--- STEP 6 ({split_name.upper()}): SEQUENCE FEATURE EXTRACTION ---")
    print(f"Input manifest : {manifest_csv}")
    print(f"Output folder  : {feat_dir}")

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name}"):
            img_path_str = resolve_image_path(row, df.columns)
            if img_path_str is None:
                failed += 1
                continue

            img_path = Path(img_path_str)
            pil_img = safe_read_gray(img_path)

            if pil_img is None:
                failed += 1
                continue

            tensor = preprocess(pil_img).unsqueeze(0).to(device)

            # Spatial feature map: (1, 2048, H, W)
            feat_map = model(tensor)

            # Flatten spatial map into sequence: (H*W, 2048)
            feat_seq = feature_map_to_sequence(feat_map)

            vec = feat_seq.detach().cpu().numpy().astype(np.float32)

            sample_name = resolve_sample_name(row, df.columns)
            feat_name = f"{sample_name}_f.npy"
            feat_path = feat_dir / feat_name
            np.save(feat_path, vec)

            seq_len = int(vec.shape[0])
            feat_dim = int(vec.shape[1])

            seq_lens.append(seq_len)

            out_row = {
                "line_id": str(row["line_id"]) if "line_id" in df.columns else Path(row["filename"]).stem,
                "filename": str(row["filename"]),
                "variant": str(row["variant"]) if "variant" in df.columns else "orig",
                "transcription": str(row["transcription"]),
                "image_path": str(img_path.resolve()),
                "feature_path": str(feat_path.resolve()),
                "seq_len": seq_len,
                "feature_dim": feat_dim
            }

            # preserve useful metadata if present
            for optional_col in [
                "input_path", "binarized_path", "denoised_path",
                "deskewed_path", "width", "height",
                "skew_angle_deg", "augmented"
            ]:
                if optional_col in df.columns:
                    out_row[optional_col] = row[optional_col]

            rows.append(out_row)

    out_manifest = out_dir / split_name / f"step6_{split_name}_features_manifest.csv"
    pd.DataFrame(rows).to_csv(out_manifest, index=False, encoding="utf-8-sig")

    print("\n------------------------------")
    print(f"Split                      : {split_name}")
    print(f"Total samples in manifest  : {len(df)}")
    print(f"Features extracted         : {len(rows)}")
    print(f"Failed                     : {failed}")
    if len(rows) > 0:
        print(f"Feature sequence length    : min={min(seq_lens)}, max={max(seq_lens)}")
        print(f"Feature dimension          : {rows[0]['feature_dim']}")
    print(f"✔ Saved manifest           : {out_manifest}")

    return out_manifest, seq_lens



# MAIN

def main():
    
    print(" STEP 6 : SEQUENCE FEATURE EXTRACTION (ResNet-50)")
    

    base = Path.cwd()
    device = get_device()
    print(f"Using device: {device}")

    train_manifest = base / "STEP5_OUTPUTS" / "splits" / "train_lines_aug.csv"
    val_manifest   = base / "STEP5_OUTPUTS" / "splits" / "val_lines.csv"
    test_manifest  = base / "STEP5_OUTPUTS" / "splits" / "test_lines.csv"

    for p in [train_manifest, val_manifest, test_manifest]:
        if not p.exists():
            print(f"ERROR: Missing file: {p}")
            return

    out_dir = base / "STEP6_OUTPUTS_SEQ"
    out_dir.mkdir(exist_ok=True)

    train_out, train_seq_lens = extract_features_for_manifest(train_manifest, out_dir, "train", device)
    val_out, val_seq_lens     = extract_features_for_manifest(val_manifest, out_dir, "val", device)
    test_out, test_seq_lens   = extract_features_for_manifest(test_manifest, out_dir, "test", device)

    all_seq_lens = train_seq_lens + val_seq_lens + test_seq_lens

    cfg = {
        "feature_dim": D_FEATURE,
        "feature_type": "sequence",
        "sequence_mode": "flatten_hw",
        "train_features_manifest": str(train_out),
        "val_features_manifest": str(val_out),
        "test_features_manifest": str(test_out),
        "train_input_manifest": str(train_manifest),
        "val_input_manifest": str(val_manifest),
        "test_input_manifest": str(test_manifest),
        "seq_len_min": int(min(all_seq_lens)) if len(all_seq_lens) > 0 else None,
        "seq_len_max": int(max(all_seq_lens)) if len(all_seq_lens) > 0 else None
    }

    cfg_path = out_dir / "step6_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("\nOutputs saved:")
    print(f"✔ Train features folder   : {out_dir / 'train' / 'features'}")
    print(f"✔ Val features folder     : {out_dir / 'val' / 'features'}")
    print(f"✔ Test features folder    : {out_dir / 'test' / 'features'}")
    print(f"✔ Config file             : {cfg_path}")

    if len(all_seq_lens) > 0:
        print(f"\nOverall sequence length range: min={min(all_seq_lens)}, max={max(all_seq_lens)}")

   


if __name__ == "__main__":
    main()