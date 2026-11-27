"""
Semantic Encoder — SBERT wrapper with caching and precomputed embedding support.
Encodes instruction texts into dense embeddings for policy conditioning.

If config/precomputed_embeddings.npz exists, all known texts are served from
that file without loading the SBERT model, saving ~500 MB of RAM.
Unknown texts fall back to the SBERT model (or placeholder if model unavailable).
"""

import os
import numpy as np
import hashlib
from typing import Dict, List, Union


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticEncoder:
    def __init__(self, model_name=DEFAULT_MODEL,
                 embedding_dim=384, device='cpu',
                 precomputed_path: str = None):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.cache: Dict[str, np.ndarray] = {}
        self.model = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._model_init_attempted = False

        # Try to load precomputed embeddings first (no SBERT needed for known texts)
        self._load_precomputed(precomputed_path)

        # Only load SBERT eagerly when no precomputed cache is available
        if not self.cache:
            self._init_model()

    def _load_precomputed(self, precomputed_path: str = None):
        """Load pre-encoded embeddings from .npz file into cache."""
        if not precomputed_path and self.model_name != DEFAULT_MODEL:
            return
        path = precomputed_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "precomputed_embeddings.npz"
        )
        if not os.path.isfile(path):
            return
        with np.load(path, allow_pickle=False) as data:
            texts = data["texts"]
            embeddings = data["embeddings"]
        if (embeddings.ndim != 2 or not embeddings.shape[1]
                or texts.ndim != 1 or len(texts) != len(embeddings)
                or not np.isfinite(embeddings).all()):
            raise ValueError(f"Invalid embedding cache: {path}")
        self.cache.update(zip(texts.tolist(), embeddings.astype(np.float32)))
        self.embedding_dim = embeddings.shape[1]
        print(f"[SemanticEncoder] Loaded {len(self.cache)} precomputed embeddings "
              f"(dim={self.embedding_dim}) from {path}")

    def _init_model(self):
        if self._model_init_attempted:
            return
        self._model_init_attempted = True
        snapshot = os.environ.get('DTMW_SBERT_SNAPSHOT')
        cache_folder = (
            os.environ.get('SENTENCE_TRANSFORMERS_HOME')
            or os.environ.get('HF_HOME')
            or os.environ.get('TRANSFORMERS_CACHE')
        )
        target = self.device
        if target.startswith('cuda'):
            import torch
            if not torch.cuda.is_available():
                target = 'cpu'

        try:
            from sentence_transformers import SentenceTransformer
            if snapshot and os.path.isdir(snapshot):
                print(f"[SemanticEncoder] Loading SBERT from local snapshot: {snapshot}")
                self.model = SentenceTransformer(snapshot)
            elif cache_folder:
                print(f"[SemanticEncoder] Loading SBERT with cache folder: {cache_folder}")
                self.model = SentenceTransformer(self.model_name, cache_folder=cache_folder)
            else:
                print(f"[SemanticEncoder] Loading SBERT model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)

            self.model.to(target)
            self.device = target
            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim:
                self.embedding_dim = actual_dim
            print(f"[SemanticEncoder] Loaded successfully. Dim={self.embedding_dim}")
            return
        except Exception as sbert_err:
            try:
                from transformers import AutoTokenizer, AutoModel
                model_src = snapshot if (snapshot and os.path.isdir(snapshot)) else self.model_name
                hf_kwargs = {'cache_dir': cache_folder} if cache_folder and model_src == self.model_name else {}
                if os.environ.get('HF_HUB_OFFLINE', '0') == '1':
                    hf_kwargs['local_files_only'] = True
                self._hf_tokenizer = AutoTokenizer.from_pretrained(model_src, **hf_kwargs)
                self._hf_model = AutoModel.from_pretrained(model_src, **hf_kwargs).to(target)
                self._hf_model.eval()
                self.device = target
                hidden_size = getattr(self._hf_model.config, 'hidden_size', None)
                if hidden_size:
                    self.embedding_dim = hidden_size
                print(f"[SemanticEncoder] Loaded via transformers fallback. Dim={self.embedding_dim}")
                return
            except Exception:
                print(f"[SemanticEncoder] SBERT failed to load ({sbert_err}), using placeholder.")
                self.model = None
                self._hf_tokenizer = None
                self._hf_model = None

    def _encode_hf(self, text: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        encoded = self._hf_tokenizer(
            [text], padding=True, truncation=True, max_length=256, return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            model_output = self._hf_model(**encoded)
        token_embeddings = model_output[0]
        mask = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        emb = F.normalize(sum_embeddings / sum_mask, p=2, dim=1)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            return self._encode_single(text)
        return np.array([self._encode_single(t) for t in text])

    def _encode_single(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        if self.model is None and self._hf_model is None and not self._model_init_attempted:
            self._init_model()
        if self.model is not None:
            emb = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        elif self._hf_model is not None:
            emb = self._encode_hf(text)
        else:
            emb = self._placeholder(text)
        self.cache[text] = emb
        return emb

    def _placeholder(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], 'little')
        rng = np.random.RandomState(seed)
        emb = rng.randn(self.embedding_dim).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-6)

    def clear_cache(self):
        self.cache.clear()
