#!/usr/bin/env python3
"""Precompute scenario instruction and weather description embeddings used by DT-MW configs."""

from pathlib import Path
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_CONFIG = ROOT / 'config' / 'scenarios.json'
WEATHER_CONFIG = ROOT / 'config' / 'weather_days.json'
OUT = ROOT / 'config' / 'precomputed_embeddings.npz'
MODEL = os.environ.get('DTMW_SBERT_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')


def collect_texts() -> list[str]:
    texts: list[str] = []
    if SCENARIOS_CONFIG.exists():
        with SCENARIOS_CONFIG.open() as f:
            cfg = json.load(f)
        for s in cfg.get('scenarios', []):
            instr = s.get('instruction', '')
            if instr and instr not in texts:
                texts.append(instr)
    if WEATHER_CONFIG.exists():
        with WEATHER_CONFIG.open() as f:
            wcfg = json.load(f)
        for d in wcfg.get('days', []):
            desc = d.get('weather_description', '')
            if desc and desc not in texts:
                texts.append(desc)
    return texts


def encode_texts(texts: list[str], model_name: str) -> np.ndarray:
    snapshot = os.environ.get('DTMW_SBERT_SNAPSHOT')
    src = snapshot if (snapshot and os.path.isdir(snapshot)) else model_name
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(src)
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embs.astype(np.float32)
    except ImportError:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(src)
        model = AutoModel.from_pretrained(src)
        model.eval()
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded)
        token_embeddings = model_output[0]
        mask = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        embs = F.normalize(sum_embeddings / sum_mask, p=2, dim=1)
        return embs.cpu().numpy().astype(np.float32)


def main():
    texts = collect_texts()
    embeddings = encode_texts(texts, MODEL)
    np.savez(OUT, texts=np.array(texts), embeddings=embeddings.astype(np.float32))
    print(f'Wrote {len(texts)} embeddings of shape {embeddings.shape} to {OUT}')


if __name__ == '__main__':
    main()
