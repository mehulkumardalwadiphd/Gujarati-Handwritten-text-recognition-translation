import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


D_MODEL = 384
NHEAD = 6
NUM_LAYERS = 3
FEATURE_DIM = 2048


# ROBUST TOKEN LOADER

def load_tokens_any(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"y_target not found: {p}")

    suf = p.suffix.lower()

    if suf == ".npy":
        y = np.load(p)
        return np.asarray(y, dtype=np.int64).reshape(-1)

    if suf == ".csv":
        df = pd.read_csv(p)
        if df.shape[1] == 0:
            raise ValueError(f"Empty CSV: {p}")

        if "token_id" in df.columns:
            return df["token_id"].astype(np.int64).values.reshape(-1)

        col0 = df.columns[0]
        try:
            return df[col0].astype(np.int64).values.reshape(-1)
        except Exception:
            y2 = np.loadtxt(p, delimiter=",", dtype=np.int64)
            return np.asarray(y2, dtype=np.int64).reshape(-1)

    raise ValueError(f"Unsupported y_target extension: {p.suffix}")



# VOCAB LOADER

def load_vocab_json(vocab_path: Path) -> Any:
    txt = vocab_path.read_text(encoding="utf-8", errors="replace")
    return json.loads(txt)


def _is_int_dict(d: dict) -> bool:
    return isinstance(d, dict) and len(d) > 0 and all(isinstance(v, int) for v in d.values())


def _is_str_dict_keys(d: dict) -> bool:
    return isinstance(d, dict) and all(isinstance(k, str) for k in d.keys())


def _is_digit_keys_dict(d: dict) -> bool:
    return isinstance(d, dict) and all(isinstance(k, str) and k.isdigit() for k in d.keys())


def normalize_vocab(vocab_obj: Any) -> Tuple[Dict[str, int], List[str]]:
    if isinstance(vocab_obj, list) and all(isinstance(x, str) for x in vocab_obj):
        itos = vocab_obj
        stoi = {tok: i for i, tok in enumerate(itos)}
        return stoi, itos

    if not isinstance(vocab_obj, dict):
        raise ValueError(f"Unsupported vocab.json root type: {type(vocab_obj)}")

    v = vocab_obj

    if "stoi" in v and "itos" in v and isinstance(v["stoi"], dict) and isinstance(v["itos"], list):
        return v["stoi"], v["itos"]

    if _is_int_dict(v) and _is_str_dict_keys(v):
        stoi = v
        itos = [""] * (max(stoi.values()) + 1)
        for tok, idx in stoi.items():
            if 0 <= idx < len(itos):
                itos[idx] = tok
        return stoi, itos

    if _is_digit_keys_dict(v):
        max_id = max(int(k) for k in v.keys())
        itos = [""] * (max_id + 1)
        for k, tok in v.items():
            itos[int(k)] = str(tok)
        stoi = {tok: i for i, tok in enumerate(itos)}
        return stoi, itos

    raise ValueError("Unsupported vocab.json format.")


def decode_tokens(token_ids: List[int], itos: List[str], eos_id: int, pad_id: int, sos_id: int) -> str:
    out = []
    for t in token_ids:
        if t == eos_id:
            break
        if t in (pad_id, sos_id):
            continue
        if 0 <= t < len(itos):
            out.append(itos[t])
    return "".join(out)



# CER / WER

def levenshtein(a: List, b: List) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def cer(ref: str, hyp: str) -> float:
    return levenshtein(list(ref), list(hyp)) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    return levenshtein(ref.split(), hyp.split()) / max(1, len(ref.split()))



# MODEL  
class TransformerDecoderOCR(nn.Module):
    def __init__(self, vocab_size: int, max_len: int = 256):
        super().__init__()

        self.feature_proj = nn.Linear(FEATURE_DIM, D_MODEL)
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, D_MODEL))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, NUM_LAYERS)
        self.fc_out = nn.Linear(D_MODEL, vocab_size)

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, feats: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, L, 2048)
        tgt:   (B, T)
        """
        T = tgt.shape[1]

        memory = self.feature_proj(feats)          # (B, L, D_MODEL)
        tgt_emb = self.token_emb(tgt) + self.pos_emb[:, :T]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)

        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )

        return self.fc_out(decoded)



# CHECKPOINT LOADER

def load_checkpoint_flexible(model: nn.Module, ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Checkpoint format is not supported.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected



# NO-REPEAT NGRAM HELPER

def has_repeated_ngram(tokens: List[int], next_token: int, ngram_size: int) -> bool:
    if ngram_size <= 0:
        return False

    candidate = tokens + [next_token]
    if len(candidate) < ngram_size * 2:
        return False

    new_ngram = tuple(candidate[-ngram_size:])
    seen = set()

    for i in range(len(candidate) - ngram_size):
        seen.add(tuple(candidate[i:i + ngram_size]))

    return new_ngram in seen



# BEAM SEARCH

@torch.no_grad()
def beam_search_decode(
    model: TransformerDecoderOCR,
    feat: torch.Tensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    beam_size: int = 5,
    length_penalty: float = 0.7,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    min_len: int = 1,
) -> List[int]:

    device = feat.device
    beams = [([sos_id], 0.0, False)]  # (tokens, score, ended)

    for _ in range(max_len):
        new_beams = []

        for tokens, score, ended in beams:
            if ended:
                new_beams.append((tokens, score, True))
                continue

            tgt = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(feat, tgt)
            next_logits = logits[0, -1].clone()

            # block PAD
            next_logits[pad_id] = -1e9

            # repetition penalty
            for t in set(tokens):
                next_logits[t] = next_logits[t] / repetition_penalty

            log_probs = F.log_softmax(next_logits, dim=-1)

            top_lp, top_id = torch.topk(log_probs, k=beam_size * 3)

            kept = 0
            for lp, idx in zip(top_lp.tolist(), top_id.tolist()):
                idx = int(idx)

                if idx == eos_id and len(tokens) <= min_len:
                    continue

                if has_repeated_ngram(tokens, idx, no_repeat_ngram_size):
                    continue

                new_tokens = tokens + [idx]
                ended_flag = (idx == eos_id)

                new_beams.append((new_tokens, score + float(lp), ended_flag))
                kept += 1

                if kept >= beam_size:
                    break

        if len(new_beams) == 0:
            break

        def score_fn(b):
            toks, lp, _ = b
            length = max(1, len(toks) - 1)
            return lp / (length ** length_penalty)

        new_beams.sort(key=score_fn, reverse=True)
        beams = new_beams[:beam_size]

        if all(b[2] for b in beams):
            break

    def final_score(b):
        toks, lp, _ = b
        length = max(1, len(toks) - 1)
        return lp / (length ** length_penalty)

    best = max(beams, key=final_score)
    return best[0]



# MAIN
def main():
    
    print(" STEP 9 : BEAM INFERENCE + EVALUATION (SEQUENCE VERSION) ")
    

    base = Path.cwd()

    test_csv = base / "STEP7_OUTPUTS" / "test_manifest.csv"
    vocab_path = base / "STEP7_OUTPUTS" / "vocab.json"
    config_path = base / "STEP7_OUTPUTS" / "train_config.json"
    ckpt_path = base / "STEP8_OUTPUTS_SEQ" / "best_model_seq.pt"

    for p in [test_csv, vocab_path, config_path, ckpt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    vocab_obj = load_vocab_json(vocab_path)
    stoi, itos = normalize_vocab(vocab_obj)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pad_id = int(cfg.get("pad_id", stoi.get("<PAD>", 0)))
    sos_id = int(cfg.get("sos_id", stoi.get("<SOS>", 1)))
    eos_id = int(cfg.get("eos_id", stoi.get("<EOS>", 2)))
    T = int(cfg.get("max_target_length", 256))

    vocab_size = len(itos)

    df = pd.read_csv(test_csv)
    required_cols = ["feature_path", "y_path", "filename", "transcription"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{test_csv} missing required columns: {missing}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device          : {device}")
    print(f"Test samples    : {len(df)}")
    print(f"Vocabulary size : {vocab_size}")
    print(f"PAD/SOS/EOS     : {pad_id}/{sos_id}/{eos_id}")
    print(f"Checkpoint      : {ckpt_path}")

    model = TransformerDecoderOCR(
        vocab_size=vocab_size,
        max_len=T
    )

    missing_keys, unexpected_keys = load_checkpoint_flexible(model, ckpt_path, device)
    print(f"Checkpoint loaded | missing={len(missing_keys)} unexpected={len(unexpected_keys)}")

    model.to(device)
    model.eval()

    out_dir = base / "STEP9_OUTPUTS_SEQ"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cer_list = []
    wer_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Beam decoding", ncols=100):
        feat_path = Path(str(row["feature_path"]))
        y_path = Path(str(row["y_path"]))

        feats = np.load(feat_path).astype(np.float32)

        if feats.ndim == 1:
            feats = feats[np.newaxis, :]

        if feats.ndim != 2 or feats.shape[1] != FEATURE_DIM:
            raise ValueError(f"Invalid feature shape {feats.shape} in {feat_path}")

        feat_t = torch.from_numpy(feats).unsqueeze(0).to(device)  # (1, L, 2048)

        gt_tokens = load_tokens_any(str(y_path))
        gt_tokens = [int(t) for t in gt_tokens.tolist() if int(t) != pad_id]
        gt_text = decode_tokens(gt_tokens, itos=itos, eos_id=eos_id, pad_id=pad_id, sos_id=sos_id)

        pred_ids = beam_search_decode(
            model=model,
            feat=feat_t,
            sos_id=sos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_len=T,
            beam_size=5,
            length_penalty=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            min_len=1
        )
        pred_text = decode_tokens(pred_ids, itos=itos, eos_id=eos_id, pad_id=pad_id, sos_id=sos_id)

        c = cer(gt_text, pred_text)
        w = wer(gt_text, pred_text)

        cer_list.append(c)
        wer_list.append(w)

        out_row = {
            "line_id": row["line_id"] if "line_id" in df.columns else Path(str(row["filename"])).stem,
            "filename": row["filename"],
            "variant": row["variant"] if "variant" in df.columns else "orig",
            "transcription": row["transcription"],
            "gt_text": gt_text,
            "pred_text": pred_text,
            "CER": c,
            "WER": w,
            "feature_path": str(feat_path.resolve()),
            "y_path": str(y_path.resolve())
        }

        for optional_col in ["image_path", "deskewed_path", "width", "height", "skew_angle_deg", "seq_len"]:
            if optional_col in df.columns:
                out_row[optional_col] = row[optional_col]

        rows.append(out_row)

    df_out = pd.DataFrame(rows)

    mean_cer = float(np.mean(cer_list)) if cer_list else float("nan")
    mean_wer = float(np.mean(wer_list)) if wer_list else float("nan")

    results_csv = out_dir / "test_predictions_seq.csv"
    summary_json = out_dir / "evaluation_summary_seq.json"

    df_out.to_csv(results_csv, index=False, encoding="utf-8-sig")

    summary = {
        "num_test_samples": int(len(df_out)),
        "mean_CER": mean_cer,
        "mean_WER": mean_wer,
        "beam_size": 5,
        "length_penalty": 0.7,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "max_target_length": T,
        "checkpoint_path": str(ckpt_path),
        "test_manifest": str(test_csv),
        "vocab_path": str(vocab_path),
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    
    print(f"Mean CER : {mean_cer:.4f}")
    print(f"Mean WER : {mean_wer:.4f}")
    print(f"Results  : {results_csv}")
    print(f"Summary  : {summary_json}")

    print("\npredictions:")
    if not df_out.empty:
        print(df_out[["filename", "gt_text", "pred_text", "CER", "WER"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()