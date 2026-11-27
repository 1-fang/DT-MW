"""
UnifiedManager — single high-level Manager interface for DT-MW (paper-aligned).

Design principles (matching the DT-MW paper):
  - ONE unified Manager produces instruction T_t and semantic dual prior
  - Input is an external semantic context dict X_t, not just env_stats
  - Supports two backends (selectable at runtime):
      1. ScenarioPromptBackend  — no API, maps weather context -> preset scenario
      2. LLMAPIBackend          — calls GPT-4o-mini, fully open-ended reasoning
  - Both backends implement the same interface and return the same output dict

Output contract (manager.step returns this dict):
    {
        "instruction_text": str,       # natural-language strategy instruction
        "instruction_emb":  np.ndarray, # SBERT embedding of instruction (384-d)
        "lambda_prior":     List[float], # [lambda_orr, lambda_fair]
        "rho_min":          float,      # semantic service threshold prior
        "phi_min":          float,      # semantic fairness threshold prior
        "beta":             List[float], # backward-compatible prior beta
        "label":            int,        # scenario index for contrastive learning
    }
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Optional


def _lambda_to_beta(lambda_vec, max_lambda: float = 20.0) -> List[float]:
    lam = np.asarray(lambda_vec, dtype=np.float32).reshape(-1)
    if lam.size != 2:
        lam = np.array([1.0, 0.5], dtype=np.float32)
    lam = np.clip(lam, 0.0, max_lambda)
    raw = np.array([1.0, lam[0], lam[1]], dtype=np.float32)
    raw = raw / raw.sum()
    return [float(x) for x in raw]


def _beta_to_lambda(beta, max_lambda: float = 20.0) -> List[float]:
    b = np.asarray(beta, dtype=np.float32).reshape(-1)
    if b.size != 3:
        b = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    b = np.clip(b, 1e-4, 1.0)
    b = b / b.sum()
    lam = np.array([b[1] / b[0], b[2] / b[0]], dtype=np.float32)
    lam = np.clip(lam, 0.0, max_lambda)
    return [float(x) for x in lam]


def _thresholds_from_beta(beta) -> tuple:
    b = np.asarray(beta, dtype=np.float32).reshape(-1)
    if b.size != 3:
        b = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    b = np.clip(b, 0.0, 1.0)
    b = b / b.sum() if b.sum() > 0 else np.array([0.4, 0.4, 0.2], dtype=np.float32)
    rho_min = float(np.clip(0.45 + 0.45 * b[1], 0.45, 0.90))
    phi_min = float(np.clip(0.45 + 0.35 * b[2], 0.45, 0.80))
    return rho_min, phi_min


# ---------------------------------------------------------------------------
# Backend 1: ScenarioPromptBackend
# No API call. Maps external context (weather_text + env_stats) to one of
# the preset scenarios defined in scenarios.json.
# ---------------------------------------------------------------------------

class ScenarioPromptBackend:
    """
    Scenario-prompt backend: looks up a preset scenario based on context.

    This is NOT a simple static scene-template switcher.
    The lookup is driven by the semantic content of the context dict:
      - weather_text / event_text are matched to scenario keywords
      - env_stats provide fallback disambiguation
    This ensures the Manager "reads" the external context X_t, not just
    switches based on internal env counters.
    """

    def __init__(self, scenarios: List[Dict], sbert):
        self.scenarios = scenarios
        self.sbert = sbert
        # Pre-encode all scenario instructions for embedding lookup
        texts = [s['instruction'] for s in scenarios]
        self._scenario_embs = self.sbert.encode(texts)  # (K, 384)
        self._current_idx = 0

    # ---- keyword matching on external context ----
    def _select_scenario_from_context(self, context: Dict) -> int:
        """
        Map context dict to the best-matching scenario index.

        Looks at:
          - context['weather_text']  (primary signal, from weather_days.json)
          - context['event_text']
          - context['env_stats']     (fallback: supply-demand ratio)
        """
        weather = (context.get('weather_text', '') or '').lower()
        event   = (context.get('event_text',   '') or '').lower()
        combined = weather + ' ' + event

        # Keyword priority order: blizzard > surge/vip > normal
        blizzard_kw = {'blizzard', 'snowstorm', 'snow', 'storm', 'emergency',
                       'road closed', 'state of emergency'}
        vip_kw      = {'vip', 'premium', 'luxury', 'corporate', 'high-value',
                       'revenue maximization'}
        surge_kw    = {'surge', 'high demand', 'weekend', 'indoor events',
                       'supply shortage', 'order volume'}

        if any(kw in combined for kw in blizzard_kw):
            return self._find_scenario_label('blizzard')
        if any(kw in combined for kw in vip_kw):
            return self._find_scenario_label('vip')
        if any(kw in combined for kw in surge_kw):
            return self._find_scenario_label('surge')

        # Fallback to env_stats supply-demand ratio
        stats = context.get('env_stats', {})
        drivers = float(stats.get('total_drivers', 2000))
        orders  = float(stats.get('total_orders',  1000))
        sd_ratio = drivers / max(orders, 1)
        if sd_ratio < 1.0:
            return self._find_scenario_label('surge')

        return self._find_scenario_label('normal')

    def _find_scenario_label(self, name: str) -> int:
        """Return index of scenario whose 'name' field matches."""
        for i, s in enumerate(self.scenarios):
            if s.get('name', '').lower() == name.lower():
                return i
        return 0  # default to first (normal)

    def step(self, context: Dict, step_t: int) -> Dict:
        idx = self._select_scenario_from_context(context)
        self._current_idx = idx
        s = self.scenarios[idx]
        instr_text = s['instruction']
        beta = list(s['beta'])
        lambda_prior = list(s.get('lambda_prior', _beta_to_lambda(beta)))
        rho_min = float(s.get('rho_min', _thresholds_from_beta(beta)[0]))
        phi_min = float(s.get('phi_min', _thresholds_from_beta(beta)[1]))
        label = int(s.get('label', idx))
        instr_emb = self._scenario_embs[idx]  # pre-computed, fast
        return {
            'instruction_text': instr_text,
            'instruction_emb':  instr_emb,
            'lambda_prior':     lambda_prior,
            'rho_min':          rho_min,
            'phi_min':          phi_min,
            'beta':             beta,
            'label':            label,
        }

    def random_switch(self) -> Dict:
        """Randomly sample a different scenario (multi-scenario training)."""
        candidates = [i for i in range(len(self.scenarios))
                      if i != self._current_idx]
        idx = random.choice(candidates) if candidates else 0
        self._current_idx = idx
        s = self.scenarios[idx]
        instr_text = s['instruction']
        beta = list(s['beta'])
        lambda_prior = list(s.get('lambda_prior', _beta_to_lambda(beta)))
        rho_min = float(s.get('rho_min', _thresholds_from_beta(beta)[0]))
        phi_min = float(s.get('phi_min', _thresholds_from_beta(beta)[1]))
        label = int(s.get('label', idx))
        instr_emb = self._scenario_embs[idx]
        return {
            'instruction_text': instr_text,
            'instruction_emb':  instr_emb,
            'lambda_prior':     lambda_prior,
            'rho_min':          rho_min,
            'phi_min':          phi_min,
            'beta':             beta,
            'label':            label,
        }

    def get_all_embeddings(self) -> np.ndarray:
        return self._scenario_embs


# ---------------------------------------------------------------------------
# Backend 2: LLMAPIBackend
# Calls an OpenAI-compatible model. Fully open-ended: instruction and dual prior
# are generated
# by the LLM from the context, not looked up from a preset table.
# ---------------------------------------------------------------------------

class LLMAPIBackend:
    """
    LLM API backend: generates instruction + lambda_prior from context.

    The LLM output is a semantic prior only. A downstream DualController
    corrects lambda using real ORR/Fairness feedback.
    Falls back to ScenarioPromptBackend on API failure.
    """

    def __init__(self, scenarios: List[Dict], sbert, llm_config: Optional[Dict] = None):
        self._fallback = ScenarioPromptBackend(scenarios, sbert)
        self.sbert = sbert
        self._llm_config = llm_config or {}
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = (self._llm_config.get('api_key') or
                           os.getenv('OPENAI_API_KEY'))
                base_url = (self._llm_config.get('base_url') or
                            os.getenv('OPENAI_BASE_URL'))
                self._client = OpenAI(api_key=api_key, base_url=base_url)
            except Exception as e:
                print(f"[LLMAPIBackend] Failed to init client: {e}. Using fallback.")
        return self._client

    def step(self, context: Dict, step_t: int) -> Dict:
        client = self._get_client()
        if client is None:
            return self._fallback.step(context, step_t)

        model = self._llm_config.get('model') or os.getenv('DTMW_MODEL', 'gpt-4o-mini')
        stats = context.get('env_stats', {})
        hour = int((step_t * stats.get('dispatch_interval', 10)) // 60) % 24
        orr  = float(stats.get('avg_orr', 0.5))
        sd_ratio = float(stats.get('sd_ratio', 1.0))

        prompt = f"""You are the cognitive manager of a city-scale ride-hailing platform.

External context at time {hour}:00:
- Weather: {context.get('weather_text', 'Clear')}
- Event: {context.get('event_text', 'None')}
- Traffic: {context.get('traffic_text', 'Normal')}
- Fleet status: {stats.get('total_drivers', '?')} idle drivers, \
{stats.get('total_orders', '?')} pending orders
- Current ORR: {orr:.1%}, Supply/Demand ratio: {sd_ratio:.2f}

Your task: generate a high-level dispatching instruction and semantic priors
for a downstream dual controller.

Return ONLY a JSON object (no extra text):
{{
  "instruction": "<1-2 sentence dispatch strategy>",
  "lambda_prior": [<lambda_orr>, <lambda_fair>],
  "thresholds": {{"rho_min": <service_threshold>, "phi_min": <fairness_threshold>}}
}}
Constraints:
- lambda_prior values must be non-negative.
- rho_min and phi_min must be in [0, 1].
- Do NOT output final beta; beta will be computed after feedback correction."""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system',
                     'content': 'You are an expert ride-hailing fleet manager. Output valid JSON only.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.7,
                max_tokens=200,
                timeout=float(os.getenv('DTMW_LLM_TIMEOUT', '20')),
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            instr_text = data.get('instruction', '')
            lambda_prior = data.get('lambda_prior', data.get('lambda', []))
            thresholds = data.get('thresholds', {}) or {}
            rho_min = data.get('rho_min', thresholds.get('rho_min', None))
            phi_min = data.get('phi_min', thresholds.get('phi_min', None))

            if (len(lambda_prior) == 2
                    and all(isinstance(x, (int, float)) for x in lambda_prior)
                    and instr_text):
                lambda_prior = [max(0.0, float(x)) for x in lambda_prior]
                beta = _lambda_to_beta(lambda_prior)
                if rho_min is None or phi_min is None:
                    rho_min, phi_min = _thresholds_from_beta(beta)
                rho_min = float(np.clip(rho_min, 0.0, 1.0))
                phi_min = float(np.clip(phi_min, 0.0, 1.0))
                instr_emb = self.sbert.encode(instr_text)
                label = self._fallback._find_scenario_label(
                    self._infer_scenario_name(beta))
                return {
                    'instruction_text': instr_text,
                    'instruction_emb':  instr_emb,
                    'lambda_prior':     lambda_prior,
                    'rho_min':          rho_min,
                    'phi_min':          phi_min,
                    'beta':             beta,
                    'label':            label,
                }

            # Backward-compatible parser if a legacy prompt/backend returns beta.
            beta = data.get('beta', [])
            if (len(beta) == 3 and all(isinstance(b, (int, float)) for b in beta)
                    and abs(sum(beta) - 1.0) < 0.1 and instr_text):
                beta = [float(b) for b in beta]
                lambda_prior = _beta_to_lambda(beta)
                rho_min, phi_min = _thresholds_from_beta(beta)
                instr_emb = self.sbert.encode(instr_text)
                label = self._fallback._find_scenario_label(
                    self._infer_scenario_name(beta))
                return {
                    'instruction_text': instr_text,
                    'instruction_emb':  instr_emb,
                    'lambda_prior':     lambda_prior,
                    'rho_min':          rho_min,
                    'phi_min':          phi_min,
                    'beta':             beta,
                    'label':            label,
                }
            else:
                print(f"[LLMAPIBackend] Bad response: {data}. Fallback.")
        except Exception as e:
            print(f"[LLMAPIBackend] API error: {e}. Fallback.")

        return self._fallback.step(context, step_t)

    def random_switch(self) -> Dict:
        return self._fallback.random_switch()

    def get_all_embeddings(self) -> np.ndarray:
        return self._fallback.get_all_embeddings()

    def _infer_scenario_name(self, beta: List[float]) -> str:
        b_gmv, b_orr, b_fair = beta
        if b_orr >= 0.5:
            return 'blizzard'
        if b_gmv >= 0.6:
            return 'vip' if b_gmv >= 0.7 else 'surge'
        return 'normal'


# ---------------------------------------------------------------------------
# UnifiedManager — the ONLY high-level interface used by train.py
# ---------------------------------------------------------------------------

class UnifiedManager:
    """
    Paper-aligned unified Manager.

    A single entry point that wraps either ScenarioPromptBackend or LLMAPIBackend.
    Always returns the same output dict so the training loop never needs to know
    which backend is active.

    Lifecycle:
        manager = UnifiedManager(scenarios_path, sbert, use_llm=False)
        # per step:
        out = manager.step(context, step_t)
        # out["instruction_text"], out["instruction_emb"], out["lambda_prior"], ...
        # random switch (multi-scenario training):
        out = manager.random_switch()
        # all scenario embeddings for contrastive learning:
        embs = manager.get_all_embeddings()  # (K, sbert_dim)
    """

    def __init__(self, scenarios_path: str, sbert,
                 use_llm: bool = False,
                 llm_config: Optional[Dict] = None,
                 update_interval: int = 3):
        """
        Args:
            scenarios_path:   path to scenarios.json
            sbert:            SemanticEncoder instance (for embedding instructions)
            use_llm:          True → LLMAPIBackend; False → ScenarioPromptBackend
            llm_config:       dict with api_key / model / base_url (LLM mode only)
            update_interval:  how many env steps between Manager updates
        """
        with open(scenarios_path, 'r') as f:
            cfg = json.load(f)
        self.scenarios = cfg['scenarios']
        self.update_interval = update_interval
        self._use_llm = use_llm

        if use_llm:
            self._backend = LLMAPIBackend(self.scenarios, sbert, llm_config)
        else:
            self._backend = ScenarioPromptBackend(self.scenarios, sbert)

        # Cache last output so callers can always read current state
        self._last_output: Optional[Dict] = None

    def build_context(self,
                      weather_text: str = '',
                      event_text: str = '',
                      traffic_text: str = '',
                      time_slot: int = 0,
                      env_stats: Optional[Dict] = None) -> Dict:
        """
        Helper: build the context dict X_t from available signals.

        This is the canonical way to construct the Manager's input.
        The training loop should call this once per step rather than
        assembling the dict inline.
        """
        return {
            'weather_text': weather_text,
            'event_text':   event_text,
            'traffic_text': traffic_text,
            'time_slot':    time_slot,
            'env_stats':    env_stats or {},
        }

    def step(self, context: Dict, step_t: int) -> Dict:
        """
        Called once every `update_interval` env steps.

        Args:
            context:  dict built via build_context()
            step_t:   current env timestep (0-indexed)

        Returns:
            dict with keys: instruction_text, instruction_emb,
            lambda_prior, rho_min, phi_min, beta, label
        """
        out = self._backend.step(context, step_t)
        self._last_output = out
        return out

    def random_switch(self) -> Dict:
        """
        Randomly switch to a different scenario.
        Used during multi-scenario training to expose the Worker to diverse β.
        """
        out = self._backend.random_switch()
        self._last_output = out
        return out

    def get_all_embeddings(self) -> np.ndarray:
        """Return (K, sbert_dim) array of all scenario instruction embeddings."""
        return self._backend.get_all_embeddings()

    @property
    def last_output(self) -> Optional[Dict]:
        """The most recently returned Manager output dict."""
        return self._last_output
