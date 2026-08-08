import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# SETTINGS

MAX_T = 256



# TEXT HELPERS

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def build_char_vocab(texts):
    vocab = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3
    }

    idx = 4
    chars = set()

    for text in texts:
        for ch in text:
            chars.add(ch)

    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1

    return vocab


def encode_text(text, vocab, max_len=MAX_T):
    ids = [vocab["<SOS>"]]

    for ch in text:
        ids.append(vocab.get(ch, vocab["<UNK>"]))

    ids.append(vocab["<EOS>"])

    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = vocab["<EOS>"]

    pad_id = vocab["<PAD>"]
    ids = ids + [pad_id] * (max_len - len(ids))
    return ids


def non_pad_length(token_ids, pad_id=0):
    return int(sum(1 for x in token_ids if x != pad_id))


# VALIDATION

def load_manifest(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} manifest not found: {path}")

    df = pd.read_csv(path)

    required_cols = ["filename", "transcription", "feature_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{split_name}] Missing required columns {missing} in {path.name}. "
            f"Found: {df.columns.tolist()}"
        )

    if "variant" not in df.columns:
        df["variant"] = "orig"

    if "line_id" not in df.columns:
        df["line_id"] = df["filename"].astype(str).apply(lambda x: Path(x).stem)

    df["split"] = split_name
    df["transcription"] = df["transcription"].apply(clean_text)

    return df


# TOKEN FILE WRITER

def save_token_file(token_ids, out_path: Path):
    pd.DataFrame({"token_id": token_ids}).to_csv(out_path, index=False, encoding="utf-8-sig")



# PROCESS ONE SPLIT

def process_split(df: pd.DataFrame, split_name: str, vocab: dict, out_root: Path):
    y_dir = out_root / split_name / "y_targets"
    y_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    max_non_pad_len = 0
    skipped_empty = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Tokenizing {split_name}"):
        text = clean_text(row["transcription"])

        if len(text) == 0:
            skipped_empty += 1
            continue

        token_ids = encode_text(text, vocab, MAX_T)
        seq_len = non_pad_length(token_ids, vocab["<PAD>"])
        max_non_pad_len = max(max_non_pad_len, seq_len)

        filename = str(row["filename"])
        line_id = str(row["line_id"])
        variant = str(row["variant"])

        sample_stem = f"{Path(filename).stem}_{variant}"
        y_path = y_dir / f"{sample_stem}_y.csv"
        save_token_file(token_ids, y_path)

        out_row = {
            "line_id": line_id,
            "filename": filename,
            "variant": variant,
            "split": split_name,
            "transcription": text,
            "feature_path": str(row["feature_path"]),
            "y_path": str(y_path.resolve()),
            "target_length": seq_len,
            "feature_dim": int(row["feature_dim"]) if "feature_dim" in row and pd.notna(row["feature_dim"]) else 2048
        }

        # preserve useful metadata
        for optional_col in [
            "image_path", "input_path", "binarized_path", "denoised_path",
            "deskewed_path", "width", "height", "skew_angle_deg", "augmented"
        ]:
            if optional_col in df.columns:
                out_row[optional_col] = row[optional_col]

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_manifest = out_root / f"{split_name}_manifest.csv"
    out_df.to_csv(out_manifest, index=False, encoding="utf-8-sig")

    return out_df, out_manifest, skipped_empty, max_non_pad_len



# MAIN

def main():
    
    print(" STEP 7 : VOCAB + TOKENIZATION + TRAIN-READY DATASET ")
    

    base = Path.cwd()

    train_feat_manifest = base / "STEP6_OUTPUTS_SEQ" / "train" / "step6_train_features_manifest.csv"
    val_feat_manifest   = base / "STEP6_OUTPUTS_SEQ" / "val" / "step6_val_features_manifest.csv"
    test_feat_manifest  = base / "STEP6_OUTPUTS_SEQ" / "test" / "step6_test_features_manifest.csv"

    out_root = base / "STEP7_OUTPUTS"
    out_root.mkdir(exist_ok=True)

    print("Loading Step-6 feature manifests...")
    df_train = load_manifest(train_feat_manifest, "train")
    df_val   = load_manifest(val_feat_manifest, "val")
    df_test  = load_manifest(test_feat_manifest, "test")

    
    # Build vocab from TRAIN only
    
    train_texts = df_train["transcription"].dropna().astype(str).apply(clean_text)
    train_texts = train_texts[train_texts.str.len() > 0].tolist()

    if len(train_texts) == 0:
        print("ERROR: No usable train transcriptions found.")
        return

    vocab = build_char_vocab(train_texts)
    vocab_path = out_root / "vocab.json"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Train texts used for vocab : {len(train_texts)}")
    print(f"Vocabulary size            : {len(vocab)}")

    
    # Tokenize all splits and build final manifests
    
    print("\nBuilding train-ready manifests...")
    train_out, train_manifest_path, train_skipped, train_max_len = process_split(df_train, "train", vocab, out_root)
    val_out, val_manifest_path, val_skipped, val_max_len = process_split(df_val, "val", vocab, out_root)
    test_out, test_manifest_path, test_skipped, test_max_len = process_split(df_test, "test", vocab, out_root)

    global_max_len = max(train_max_len, val_max_len, test_max_len)

    
    # Save final training config
    
    config = {
        "vocab_size": int(len(vocab)),
        "pad_id": int(vocab["<PAD>"]),
        "sos_id": int(vocab["<SOS>"]),
        "eos_id": int(vocab["<EOS>"]),
        "unk_id": int(vocab["<UNK>"]),
        "feature_dim": 2048,
        "max_target_length": int(MAX_T),
        "max_non_pad_length_seen": int(global_max_len),
        "train_samples": int(len(train_out)),
        "val_samples": int(len(val_out)),
        "test_samples": int(len(test_out)),
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "test_manifest": str(test_manifest_path),
        "vocab_path": str(vocab_path)
    }

    config_path = out_root / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

   
    # Final report
    
    print("\n------------------------------")
    print("Final summary:")
    print(f"Train samples kept         : {len(train_out)}")
    print(f"Validation samples kept    : {len(val_out)}")
    print(f"Test samples kept          : {len(test_out)}")
    print(f"Skipped empty train rows   : {train_skipped}")
    print(f"Skipped empty val rows     : {val_skipped}")
    print(f"Skipped empty test rows    : {test_skipped}")
    print(f"Vocabulary size            : {len(vocab)}")
    print(f"Max target length (MAX_T)  : {MAX_T}")
    print(f"Max non-pad length seen    : {global_max_len}")

    print("\nOutputs saved:")
    print(f" Vocab file              : {vocab_path}")
    print(f" Train manifest          : {train_manifest_path}")
    print(f" Validation manifest     : {val_manifest_path}")
    print(f" Test manifest           : {test_manifest_path}")
    print(f" Training config         : {config_path}")
    print(f" Train y targets folder  : {out_root / 'train' / 'y_targets'}")
    print(f" Val y targets folder    : {out_root / 'val' / 'y_targets'}")
    print(f" Test y targets folder   : {out_root / 'test' / 'y_targets'}")

    if not train_out.empty:
        print("\nSample train-ready rows:")
        print(train_out[["filename", "variant", "transcription", "feature_path", "y_path"]].head(5).to_string(index=False))

    


if __name__ == "__main__":
    main()