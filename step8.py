import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




SEED = 42
BATCH_SIZE = 32
EPOCHS = 25

LEARNING_RATE = 1e-4     
WEIGHT_DECAY = 1e-4
PATIENCE = 6

D_MODEL = 384            
NHEAD = 6
NUM_LAYERS = 3
FEATURE_DIM = 2048

DROPOUT = 0.3           



# REPRODUCIBILITY

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# TOKEN LOADER

def load_tokens_any(path: str) -> np.ndarray:
    p = Path(path)

    if p.suffix.lower() == ".npy":
        return np.load(p).astype(np.int64)

    df = pd.read_csv(p)
    if "token_id" in df.columns:
        return df["token_id"].astype(np.int64).values

    return df.iloc[:, 0].astype(np.int64).values


def pad_or_trim(y, T, pad_id):
    out = np.full((T,), pad_id, dtype=np.int64)
    y = y[:T]
    out[:len(y)] = y
    return out



# DATASET

class LineFeatureDataset(Dataset):
    def __init__(self, csv_path, feat_col, y_col, T, pad_id):
        self.df = pd.read_csv(csv_path)
        self.feat_col = feat_col
        self.y_col = y_col
        self.T = T
        self.pad_id = pad_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feats = np.load(row[self.feat_col]).astype(np.float32)

        if feats.ndim == 1:
            feats = feats[np.newaxis, :]

        y = load_tokens_any(row[self.y_col])
        y = pad_or_trim(y, self.T, self.pad_id)

        return torch.from_numpy(feats), torch.from_numpy(y)



# COLLATE (PAD SEQUENCE LENGTH)

def collate_fn(batch):
    feats, ys = zip(*batch)

    max_S = max(f.shape[0] for f in feats)

    padded_feats = []
    for f in feats:
        if f.shape[0] < max_S:
            pad = torch.zeros(max_S - f.shape[0], f.shape[1])
            f = torch.cat([f, pad], dim=0)
        padded_feats.append(f)

    feats = torch.stack(padded_feats)
    ys = torch.stack(ys)

    return feats, ys



# MODEL (UPDATED)

class TransformerDecoderOCR(nn.Module):
    def __init__(self, vocab_size, max_len):
        super().__init__()

        self.feature_proj = nn.Linear(FEATURE_DIM, D_MODEL)
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, D_MODEL))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dropout=DROPOUT,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)

        self.dropout = nn.Dropout(DROPOUT)
        self.fc_out = nn.Linear(D_MODEL, vocab_size)

        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, feats, tgt_tokens):
        B, S, _ = feats.shape
        T = tgt_tokens.shape[1]

        memory = self.feature_proj(feats)

        tgt_emb = self.token_emb(tgt_tokens) + self.pos_emb[:, :T]
        tgt_emb = self.dropout(tgt_emb)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_emb.device)

        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )

        logits = self.fc_out(decoded)
        return logits



# TRAIN LOOP

def run_epoch(model, loader, criterion, optimizer, vocab_size, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0

    with torch.set_grad_enabled(train):
        for feats, y in tqdm(loader, leave=False):
            feats, y = feats.to(device), y.to(device)

            inp = y[:, :-1]
            tgt = y[:, 1:]

            logits = model(feats, inp)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt.reshape(-1)
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)

# MAIN

def main():
    
    print(" STEP 8 : TRANSFORMER TRAINING (OPTIMIZED)")
    

    set_seed(SEED)

    base = Path.cwd()

    train_csv = base / "STEP7_OUTPUTS" / "train_manifest.csv"
    val_csv   = base / "STEP7_OUTPUTS" / "val_manifest.csv"
    cfg_path  = base / "STEP7_OUTPUTS" / "train_config.json"
    vocab_path = base / "STEP7_OUTPUTS" / "vocab.json"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    vocab_size = max(vocab.values()) + 1
    pad_id = cfg["pad_id"]
    T = cfg["max_target_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Vocab  : {vocab_size}")
    print(f"T      : {T}")

    train_ds = LineFeatureDataset(train_csv, "feature_path", "y_path", T, pad_id)
    val_ds   = LineFeatureDataset(val_csv, "feature_path", "y_path", T, pad_id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = TransformerDecoderOCR(vocab_size, max_len=T).to(device)

   
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    out_dir = base / "STEP8_OUTPUTS_SEQ"
    out_dir.mkdir(exist_ok=True)

    best_path = out_dir / "best_model_seq.pt"

    best_val = float("inf")
    bad = 0

    for ep in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, vocab_size, device, True)
        vl = run_epoch(model, val_loader, criterion, optimizer, vocab_size, device, False)

        print(f"Epoch {ep:02d} | Train {tr:.4f} | Val {vl:.4f}")

        if vl < best_val:
            best_val = vl
            bad = 0
            torch.save(model.state_dict(), best_path)
            print("Saved best model")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping")
                break

  

if __name__ == "__main__":
    main()