"""
DT-MW Training Script — Closed-Loop Fusion
==========================================
Combines LLM semantic prior with dual feedback correction:

  UnifiedLLMPriorManager:  context X_t → instruction, lambda_prior, thresholds
  DualController:          real ORR/Fairness feedback → corrected lambda
  lambda_to_beta:          corrected lambda → beta used by DRS

  - instruction comes ONLY from LLM prior
  - LLM provides semantic prior, not final beta
  - beta is produced only after feedback-corrected lambda
"""

import os
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
import json
import time
import random
import argparse
from pathlib import Path
from tempfile import mkdtemp
import numpy as np
import torch
import torch.nn as nn

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run.environment import ROOT, configure_environment, create_environment, legacy_root
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print("[DTMW] TensorBoard is not available; scalar logs are disabled.")

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

# DT-MW modules
from llm.semantic_encoder import SemanticEncoder
from llm.reward_shaper import DynamicRewardShaper
from llm.unified_manager import UnifiedManager
from llm.fusion_manager import (
    UnifiedLLMPriorManager, DualController,
    beta_to_lambda, lambda_to_beta,
)
from llm_algo.mappo_llm import PPOAgent, ContrastiveHead, InstructionBuffer


# ---------------------------------------------------------------------------
# Multi-day weather data loader
# ---------------------------------------------------------------------------

def load_weather_days(weather_config_path, data_dir):
    """
    Read scenario contexts and paths. Each episode loads its orders on demand.
    """
    with open(weather_config_path, 'r') as f:
        cfg = json.load(f)

    days = []
    weights = cfg.get("sample_weights", [1.0] * len(cfg["days"]))
    if len(weights) != len(cfg["days"]) or any(not np.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Weather sample_weights must match days and contain finite nonnegative values")
    selected_weights = []
    for day_cfg, weight in zip(cfg["days"], weights):
        data_path = os.path.join(data_dir, day_cfg['data_file'])
        if not os.path.exists(data_path):
            print(f"[DTMW] WARNING: data file not found: {data_path}, skipping {day_cfg['name']}")
            continue

        days.append({
            'name': day_cfg['name'],
            'data_path': data_path,          # path only, orders loaded per-episode
            'weather_description': day_cfg['weather_description'],
            'event_description': day_cfg.get('event_description', ''),
            'traffic_description': day_cfg.get('traffic_description', ''),
            'scenario': day_cfg.get('scenario', 'normal'),
            'beta_hint': day_cfg.get('beta_hint', [0.4, 0.4, 0.2]),
        })
        selected_weights.append(weight)
        print(f"[DTMW] Registered {day_cfg['name']}: scenario={day_cfg.get('scenario','normal')}")

    if not days:
        print("[DTMW] WARNING: no weather days loaded, will use single-day mode")
    elif sum(selected_weights) <= 0:
        raise ValueError("Weather sampling needs a positive weight for at least one available day")
    return days, selected_weights


def sample_orders_from_file(data_path: str, sample_rate: float, rng=None):
    """Load one day's real_orders from disk and return a sampled list."""
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    ro = data.get('real_orders', None)
    if ro is None:
        return np.empty((0, 5), dtype=np.float32)
    if not isinstance(ro, np.ndarray):
        ro = np.asarray(ro)
    if len(ro) == 0:
        return np.empty((0, 5), dtype=np.float32)
    n_sample = max(1, int(len(ro) * sample_rate))
    if rng is None:
        idx = np.random.choice(len(ro), size=n_sample, replace=False)
    else:
        idx = rng.choice(len(ro), size=n_sample, replace=False)
    return ro[idx]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_parameter():
    parser = argparse.ArgumentParser(description="Train DT-MW. See the README for environment variables.")
    args = parser.parse_args()

    # --- Basic ---
    args.MAX_ITER = int(os.environ.get("DTMW_MAX_ITER", os.environ.get("COOPRIDE_MAX_ITER", "100")))
    args.model_save_interval = int(os.environ.get("DTMW_SAVE_INTERVAL", os.environ.get("COOPRIDE_SAVE_INTERVAL", "10")))
    args.test = os.environ.get("DTMW_TEST", os.environ.get("COOPRIDE_TEST", "0")) == "1"
    args.device = os.environ.get("DTMW_DEVICE", os.environ.get("COOPRIDE_DEVICE", "gpu"))

    # --- Environment ---
    args.dispatch_interval = 10
    args.speed = args.dispatch_interval
    args.wait_time = args.dispatch_interval
    args.TIME_LEN = int(1440 // args.dispatch_interval)
    configure_environment(args, default_grid=143)
    args.city_time_start = 0
    args.dynamic_env = False
    args.custom_data_path = os.environ.get("DTMW_DATA_PATH") or os.environ.get("COOPRIDE_DATA_PATH")
    args.use_real_orders = os.environ.get(
        "DTMW_USE_REAL_ORDERS", os.environ.get("COOPRIDE_USE_REAL_ORDERS", "0")) == "1"
    args.real_order_sample_rate = float(os.environ.get(
        "DTMW_REAL_ORDER_SAMPLE_RATE",
        os.environ.get("COOPRIDE_REAL_ORDER_SAMPLE_RATE", "0.1")))

    # --- PPO hyperparams (paper: lr=3e-4, γ=0.97, λ=0.95, ε=0.2) ---
    args.batch_size = 1000
    args.actor_lr = float(os.environ.get("DTMW_LR", "1e-3"))
    args.critic_lr = float(os.environ.get("DTMW_LR", "1e-3"))
    args.gamma = 0.97
    args.lam = 0.95
    args.max_grad_norm = 10
    args.clip_ratio = 0.2
    args.ent_factor = 0.005
    args.adv_normal = True
    args.minibatch_num = 5
    args.train_actor_iters = 1
    args.train_critic_iters = 1
    args.reward_scale = 5
    args.parallel_episode = int(os.getenv("DTMW_WARMUP_EPISODES", "5"))
    args.steps_per_epoch = args.TIME_LEN
    if args.MAX_ITER < 1 or args.parallel_episode < 1:
        parser.error("DTMW_MAX_ITER and DTMW_WARMUP_EPISODES must be positive")
    if not 0 < args.real_order_sample_rate <= 1:
        parser.error("DTMW_REAL_ORDER_SAMPLE_RATE must be in (0, 1]")
    args.new_order_entropy = True
    args.order_grid = True
    args.feature_normal = int(os.environ.get("DTMW_FEATURE_NORMAL", os.environ.get("COOPRIDE_FEATURE_NORMAL", "3")))

    # --- DT-MW specific ---
    args.use_llm = os.environ.get("DTMW_USE_LLM", "0") == "1"
    args.scenario_config = os.environ.get(
        "DTMW_SCENARIO_CONFIG",
        ROOT / 'config' / 'scenarios.json')
    args.train_weather_text = os.environ.get("DTMW_TRAIN_WEATHER_TEXT", "")
    args.train_event_text = os.environ.get("DTMW_TRAIN_EVENT_TEXT", "")
    args.train_traffic_text = os.environ.get("DTMW_TRAIN_TRAFFIC_TEXT", "")
    args.sbert_model = os.environ.get(
        "DTMW_SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    args.contrastive_weight = float(os.environ.get("DTMW_CONTRASTIVE_WEIGHT", "0.1"))
    args.intrinsic_weight = float(os.environ.get("DTMW_INTRINSIC_WEIGHT", "0.1"))
    args.contrastive_lr = float(os.environ.get("DTMW_CONTRASTIVE_LR", "1e-3"))
    args.manager_update_interval = int(os.environ.get("DTMW_MANAGER_INTERVAL", "3"))
    args.switch_probability = float(os.environ.get("DTMW_SWITCH_PROB", "0.0"))
    args.drs_weight = float(os.environ.get("DTMW_DRS_WEIGHT", "1.0"))
    args.no_ssa = os.environ.get("DTMW_NO_SSA", "0") == "1"
    args.no_manager = os.environ.get("DTMW_NO_MANAGER", "0") == "1"
    args.base_reward_orr_weight = float(os.environ.get("DTMW_BASE_ORR_WEIGHT", "1.0"))
    args.base_anchor_weight = float(os.environ.get(
        "DTMW_BASE_ANCHOR_WEIGHT", os.environ.get("DTMW4_BASE_ANCHOR_WEIGHT", "0.0")))

    # --- DT-MW: LLM prior + dual feedback controller ---
    args.use_llm_prior = os.environ.get(
        "DTMW_USE_LLM_PRIOR", os.environ.get("DTMW4_USE_LLM_PRIOR", "1")) == "1"
    args.no_dual = os.environ.get(
        "DTMW_NO_DUAL", os.environ.get("DTMW4_NO_DUAL", "0")) == "1"
    args.dual_lr = float(os.environ.get(
        "DTMW_DUAL_LR", os.environ.get("DTMW4_DUAL_LR", "0.10")))
    args.dual_prior_pull = float(os.environ.get(
        "DTMW_DUAL_PRIOR_PULL", os.environ.get("DTMW4_DUAL_PRIOR_PULL", "0.05")))
    args.dual_lambda_max = float(os.environ.get(
        "DTMW_DUAL_LAMBDA_MAX", os.environ.get("DTMW4_DUAL_LAMBDA_MAX", "20.0")))

    # --- Memory ---
    args.memory_size = int(args.TIME_LEN * args.parallel_episode)

    return args


def initialize_logging(args):
    # Build ablation tag
    ablation_parts = []
    if args.no_dual:
        ablation_parts.append('noDual')
    if not args.use_llm_prior:
        ablation_parts.append('noLLMprior')
    if args.no_ssa:
        ablation_parts.append('noSSA')
    if args.drs_weight == 0:
        ablation_parts.append('noDRS')
    if args.intrinsic_weight == 0 and args.contrastive_weight == 0:
        ablation_parts.append('noISA')
    ablation_tag = '_' + '_'.join(ablation_parts) if ablation_parts else ''
    run_tag = os.environ.get("DTMW_RUN_TAG", os.environ.get("DTMW4_RUN_TAG", "")).strip()
    run_tag = f'_{run_tag}' if run_tag else ''
    args.log_name = f'DTMW_OD{args.grid_num}_dual{ablation_tag}{run_tag}'
    output = os.getenv("DTMW_LOG_DIR")
    if output:
        log_dir = Path(output).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        parent = ROOT / "logs"
        parent.mkdir(exist_ok=True)
        log_dir = parent / args.log_name
        try:
            log_dir.mkdir()
        except FileExistsError:
            log_dir = Path(mkdtemp(prefix=f"{args.log_name}_", dir=parent))
    args.log_dir = str(log_dir)
    print(f"Log directory: {log_dir}")

    with (log_dir / "setting.txt").open("w", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(f"{key} : {value}\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_static_manager_context(scenario_config, sbert):
    scenarios = scenario_config.get('scenarios', [])
    if scenarios:
        s = scenarios[0]
        instr_text = s.get('instruction', 'Standard operation.')
        beta = list(s.get('beta', [0.4, 0.4, 0.2]))
        label = int(s.get('label', 0))
    else:
        instr_text = 'Standard operation.'
        beta = [0.4, 0.4, 0.2]
        label = 0
    instr_emb = sbert.encode(instr_text)
    return instr_text, beta, label, instr_emb


def compute_driver_fairness(env) -> float:
    """Return 1-Gini over all registered driver incomes; 0.5 when unavailable."""
    incomes = DynamicRewardShaper.collect_driver_incomes(env)
    if not incomes:
        return 0.5
    return float(1.0 - DynamicRewardShaper._gini(incomes))


def update_dual_from_feedback(dual_controller, lambda_prior, rho_min, phi_min,
                              order_response_rates, env):
    observed_orr = float(order_response_rates[-1]) if order_response_rates else 0.5
    observed_fair = compute_driver_fairness(env)
    return dual_controller.step(
        lambda_prior=lambda_prior,
        rho_min=rho_min,
        phi_min=phi_min,
        observed_orr=observed_orr,
        observed_fair=observed_fair)


def use_prior_without_dual(lambda_prior, rho_min, phi_min):
    """No-dual ablation: use LLM/static prior directly without feedback correction."""
    lambda_prior = np.asarray(lambda_prior, dtype=np.float32)
    return {
        'lambda': lambda_prior,
        'beta': lambda_to_beta(lambda_prior),
        'violation': np.asarray([0.0, 0.0], dtype=np.float32),
        'rho_min': float(rho_min),
        'phi_min': float(phi_min),
    }

def train(env, agent, writer, args, device,
          sbert, llm_prior, dual_controller,
          reward_shaper, contrastive_head, instr_buffer,
          all_scenario_embs, scenario_config,
          weather_days=None, weather_sample_weights=None):
    """DT-MW LLM-prior + dual feedback training.

    Data flow each timestep:
      1. LLM prior -> instruction_text, instruction_emb, lambda_prior, thresholds
      2. DualController -> corrected lambda using observed ORR/Fairness
      3. lambda_to_beta -> beta for DRS
      4. SSA/ISA use instruction_emb from LLM prior
      5. Worker PPO update on DRS reward
    """

    best_gmv = 0
    best_orr = 0
    use_real = getattr(args, 'use_real_orders', False)
    gen_order = 0 if use_real else 1

    # Use the env-var / CLI switch probability (args.switch_probability) so that
    # ablation scripts can override the value in scenarios.json.
    p_switch = 0.0 if args.no_manager else args.switch_probability
    eta = args.intrinsic_weight
    contrastive_w = args.contrastive_weight
    drs_weight = args.drs_weight
    base_orr_w = args.base_reward_orr_weight
    base_anchor_w = args.base_anchor_weight

    # Contrastive optimizer
    contrastive_optimizer = torch.optim.Adam(
        contrastive_head.parameters(), lr=args.contrastive_lr, eps=1e-5)

    # Pre-encode all scenario instructions
    all_instr_tensor = torch.FloatTensor(all_scenario_embs).to(device)  # (K, 384)

    # Initial manager output. Single-day datasets can provide semantic context
    # via DTMW_TRAIN_*_TEXT; multi-day weather config overwrites these per day.
    current_weather_text = args.train_weather_text
    current_event_text = args.train_event_text
    current_traffic_text = args.train_traffic_text
    current_day_name = 'default'
    # Static fallback instruction (for no_manager / no LLM-prior mode)
    _, static_beta, static_label, static_instr_emb = \
        get_static_manager_context(scenario_config, sbert)

    if args.no_manager:
        current_instr_emb = static_instr_emb
        current_label = static_label
        current_lambda_prior = beta_to_lambda(np.array(static_beta, dtype=np.float32))
        current_rho_min, current_phi_min = 0.60, 0.55
    elif not args.use_llm_prior:
        # No LLM prior: keep static instruction/prior, still allow feedback correction.
        current_instr_emb = static_instr_emb
        current_label = static_label
        current_lambda_prior = beta_to_lambda(np.array(static_beta, dtype=np.float32))
        current_rho_min, current_phi_min = 0.60, 0.55
    else:
        init_context = llm_prior.build_context(
            weather_text=current_weather_text,
            event_text=current_event_text,
            traffic_text=current_traffic_text,
            time_slot=0,
            env_stats={})
        prior_out = llm_prior.step(init_context, 0)
        current_instr_emb  = prior_out['instruction_emb']
        current_label      = prior_out['label']
        current_lambda_prior = prior_out['lambda_prior']
        current_rho_min = prior_out['rho_min']
        current_phi_min = prior_out['phi_min']

    dual_controller.reset(current_lambda_prior)
    if args.no_dual:
        current_dual = use_prior_without_dual(
            current_lambda_prior, current_rho_min, current_phi_min)
    else:
        current_dual = dual_controller.step(
            lambda_prior=current_lambda_prior,
            rho_min=current_rho_min,
            phi_min=current_phi_min,
            observed_orr=0.5,
            observed_fair=0.5)
    current_lambda = current_dual['lambda']
    current_beta = current_dual['beta'].tolist()
    current_violation = current_dual['violation']
    if not args.no_ssa:
        env.set_global_instruction_vector(current_instr_emb)

    # Keep scenario/order sampling independent from env.reset_randomseed().
    # Otherwise single-day mode seeds can collapse multi-day training to one day.
    day_rng = np.random.default_rng(args.env_seed * 1000 + 20260502)

    for iteration in range(args.MAX_ITER):
        t_begin = time.time()
        print(f'\n---- ROUND: #{iteration} ----')

        # ---- Multi-day: sample one day's data for this episode ----
        current_weather_text = args.train_weather_text
        current_event_text = args.train_event_text
        current_traffic_text = args.train_traffic_text
        current_day_name = 'default'
        if weather_days:
            probs = None
            if weather_sample_weights and len(weather_sample_weights) == len(weather_days):
                total = sum(weather_sample_weights)
                probs = [w / total for w in weather_sample_weights]
            day_idx = day_rng.choice(len(weather_days), p=probs)
            chosen_day = weather_days[day_idx]
            sampled = sample_orders_from_file(
                chosen_day['data_path'], args.real_order_sample_rate, rng=day_rng)
            env.real_orders = sampled
            env.day_orders = []  # Rebuild cached orders for the new day.
            current_weather_text = chosen_day['weather_description']
            current_event_text = chosen_day.get('event_description', '')
            current_traffic_text = chosen_day.get('traffic_description', '')
            current_day_name = chosen_day['name']
            print(f"    [Day] {current_day_name} | "
                  f"sampled={len(sampled)} | "
                  f"scenario={chosen_day['scenario']}")

        if args.dynamic_env:
            env.reset_randomseed(iteration * 1000)
        else:
            env.reset_randomseed(args.env_seed * 1000)

        gmv_list = []
        order_response_rates = []
        intrinsic_rewards_log = []

        # Reset env
        states_node, _, order_states, order_idx, order_feature, _ = \
            env.reset(mode='PPO2', generate_order=gen_order)
        state = agent.process_state(states_node, 0)
        order, mask_order = agent.process_order(order_states, 0)

        for T in range(args.TIME_LEN):
            # ---- 1. Manager prior + dual feedback correction ----
            prior_refreshed = False
            if (not args.no_manager) and args.use_llm_prior and (np.random.random() < p_switch):
                prior_out = llm_prior.random_switch()
                current_instr_emb  = prior_out['instruction_emb']
                current_lambda_prior = prior_out['lambda_prior']
                current_rho_min = prior_out['rho_min']
                current_phi_min = prior_out['phi_min']
                current_label = prior_out['label']
                prior_refreshed = True
                if not args.no_ssa:
                    env.set_global_instruction_vector(current_instr_emb)

            if T % args.manager_update_interval == 0:
                total_drivers = sum(n.idle_driver_num for n in env.nodes if n)
                total_orders  = sum(n.real_order_num  for n in env.nodes if n)
                env_stats = {
                    'total_drivers':    total_drivers,
                    'total_orders':     total_orders,
                    'sd_ratio':         total_drivers / max(total_orders, 1),
                    'avg_orr':          order_response_rates[-1] if order_response_rates else 0.5,
                    'dispatch_interval': args.dispatch_interval,
                    'step_t':           T,
                }

                if (not args.no_manager) and args.use_llm_prior and not prior_refreshed:
                    context = llm_prior.build_context(
                        weather_text=current_weather_text,
                        event_text=current_event_text,
                        traffic_text=current_traffic_text,
                        time_slot=T,
                        env_stats=env_stats,
                    )
                    prior_out = llm_prior.step(context, T)
                    current_instr_emb  = prior_out['instruction_emb']
                    current_lambda_prior = prior_out['lambda_prior']
                    current_rho_min = prior_out['rho_min']
                    current_phi_min = prior_out['phi_min']
                    current_label = prior_out['label']
                    if not args.no_ssa:
                        env.set_global_instruction_vector(current_instr_emb)

                if args.no_dual:
                    current_dual = use_prior_without_dual(
                        current_lambda_prior, current_rho_min, current_phi_min)
                else:
                    current_dual = update_dual_from_feedback(
                        dual_controller, current_lambda_prior,
                        current_rho_min, current_phi_min,
                        order_response_rates, env)
                current_lambda = current_dual['lambda']
                current_beta = current_dual['beta'].tolist()
                current_violation = current_dual['violation']

            instr_buffer.push(current_instr_emb, current_label)

            # ---- 2. RL Actor dispatch (MLP Pointer Network) ----
            action, value, logp, mask_agent, mask_order_multi, \
                mask_action, mask_entropy, action_ids, selected_ids = \
                agent.action(state, order, mask_order, order_idx, sample=True)

            orders = env.get_orders_by_id(action_ids)

            # ---- 3. Environment step ----
            next_states_node, next_order_states, next_order_idx, next_order_feature = \
                env.step(orders, generate_order=gen_order, mode='PPO2')

            gmv_list.append(env.gmv)
            if env.order_response_rate >= 0:
                order_response_rates.append(env.order_response_rate)

            # ---- 4. Reward ----
            #   base_reward_orr_weight controls how much ORR the base signal carries.
            #   When 0, ORR optimisation comes *only* from DRS β_orr.
            if hasattr(env, 'get_node_reward_vector') and env.node_reward_cache is not None:
                base_reward = torch.Tensor(env.get_node_reward_vector())
            else:
                base_reward = torch.Tensor([n.gmv if n else 0 for n in env.nodes])
            if base_orr_w < 1.0:
                # Subtract the ORR portion that the env baked in, keep only
                # the fraction requested.  env reward = gmv*0.01 + 1.0*orr
                # so orr_part ≈ node_orr.  We reconstruct it from the env.
                orr_part = torch.Tensor([
                    (min(n.idle_driver_num, n.real_order_num) / max(n.real_order_num, 1))
                    if n else 0.0 for n in env.nodes])
                base_reward = base_reward - (1.0 - base_orr_w) * orr_part
            # DRS reward uses beta from feedback-corrected lambda.
            shaped = torch.Tensor(
                reward_shaper.shape_reward(env.nodes, current_beta, env=env))
            if drs_weight == 0:
                # noDRS ablation falls back to the environment/base reward.
                reward = base_reward
            else:
                reward = drs_weight * shaped
                if base_anchor_w > 0:
                    reward = reward + base_anchor_w * base_reward

            # ---- 5. Intrinsic semantic alignment reward ----
            next_state = agent.process_state(next_states_node, T + 1)
            if eta > 0:
                try:
                    with torch.no_grad():
                        ns_emb = agent.actor.get_state_emb(next_state.to(device))
                        instr_t = torch.FloatTensor(current_instr_emb).to(device)
                        r_int = contrastive_head.intrinsic_reward(ns_emb, instr_t)
                        r_int = torch.clamp(r_int, min=0.0).cpu()
                    reward = reward + eta * r_int
                    intrinsic_rewards_log.append(r_int.mean().item())
                except Exception as e:
                    if iteration == 0 and T == 0:
                        print(f"[WARN] Intrinsic reward failed: {e}")

            reward = reward / args.reward_scale

            # ---- 6. Store transition ----
            next_order, next_mask_order = agent.process_order(next_order_states, T + 1)

            agent.buffer.push(
                state, next_state, order, action, reward[:, None],
                value, logp,
                mask_order_multi, mask_action, mask_agent, mask_entropy)

            # ---- 7. GAE at epoch/episode boundaries ----
            epoch_ended = (T % args.steps_per_epoch) == (args.steps_per_epoch - 1)
            done = T == args.TIME_LEN - 1
            if done or epoch_ended:
                if done:
                    last_value = torch.zeros(agent.agent_num)
                else:
                    with torch.no_grad():
                        last_value = agent.critic(
                            next_state.to(device)).squeeze(-1).cpu()
                agent.buffer.finish_path(last_value)

            # Advance
            states_node = next_states_node
            order_idx = next_order_idx
            order_states = next_order_states
            state = next_state
            order = next_order
            mask_order = next_mask_order

        # ---- 8. PPO + Contrastive Update ----
        # Update every iteration after warmup, matching the batched dispatch training schedule
        contrastive_losses = []
        should_update = (iteration + 1) >= args.parallel_episode
        if should_update:
            # Contrastive update first (needs buffer data)
            if contrastive_w > 0 and instr_buffer.size > 0:
                cl = _update_contrastive(
                    agent, contrastive_head, contrastive_optimizer,
                    instr_buffer, all_instr_tensor, device,
                    max_grad_norm=args.max_grad_norm)
                contrastive_losses.append(cl)
                instr_buffer.clear()
            # PPO update (clears buffer internally)
            agent.update(device, writer)

        # ---- 9. Logging ----
        t_end = time.time()
        mean_orr = np.mean(order_response_rates) if order_response_rates else 0.0
        total_gmv = np.sum(gmv_list)
        mean_fair = compute_driver_fairness(env)

        if total_gmv > best_gmv:
            best_gmv = total_gmv
            agent.save_param(args.log_dir, 'Best_GMV')
            torch.save(contrastive_head.state_dict(),
                       os.path.join(args.log_dir, 'contrastive_best_gmv.pt'))

        if mean_orr > best_orr:
            best_orr = mean_orr
            agent.save_param(args.log_dir, 'Best_ORR')
            torch.save(contrastive_head.state_dict(),
                       os.path.join(args.log_dir, 'contrastive_best_orr.pt'))

        if args.model_save_interval > 0 and (iteration + 1) % args.model_save_interval == 0:
            agent.save_param(args.log_dir, f'iter_{iteration + 1}')

        avg_cl = np.mean(contrastive_losses) if contrastive_losses else 0.0
        avg_rint = np.mean(intrinsic_rewards_log) if intrinsic_rewards_log else 0.0
        print(f'>>> Time: [{t_end - t_begin:<.4f}] ORR: [{mean_orr:<.4f}] '
              f'GMV: [{total_gmv:<.4f}] Best_ORR: [{best_orr:<.4f}] '
              f'Best_GMV: [{best_gmv:<.4f}] CL: [{avg_cl:<.4f}] '
              f'r_int: [{avg_rint:<.4f}] '
              f'lambda_prior: {[f"{x:.2f}" for x in current_lambda_prior]} '
              f'lambda: {[f"{x:.2f}" for x in current_lambda]} '
              f'beta: {[f"{b:.2f}" for b in current_beta]} '
              f'thr: [{current_rho_min:.2f},{current_phi_min:.2f}] '
              f'viol: {[f"{v:.2f}" for v in current_violation]} '
              f'Fair: [{mean_fair:.4f}] Day: {current_day_name}')

        agent.save_param(args.log_dir, 'param')
        if writer is not None:
            writer.add_scalar('train ORR', mean_orr, iteration)
            writer.add_scalar('train GMV', total_gmv, iteration)
            writer.add_scalar('train contrastive_loss', avg_cl, iteration)
            writer.add_scalar('train intrinsic_reward', avg_rint, iteration)
            writer.add_scalar('train fairness', mean_fair, iteration)
            writer.add_scalar('train beta_gmv', current_beta[0], iteration)
            writer.add_scalar('train beta_orr', current_beta[1], iteration)
            writer.add_scalar('train beta_fair', current_beta[2], iteration)
            writer.add_scalar('train lambda_orr', current_lambda[0], iteration)
            writer.add_scalar('train lambda_fair', current_lambda[1], iteration)
            writer.add_scalar('train lambda_prior_orr', current_lambda_prior[0], iteration)
            writer.add_scalar('train lambda_prior_fair', current_lambda_prior[1], iteration)
            writer.add_scalar('train violation_orr', current_violation[0], iteration)
            writer.add_scalar('train violation_fair', current_violation[1], iteration)


def _update_contrastive(agent, contrastive_head, optimizer,
                        instr_buffer, all_instr_tensor, device,
                        max_grad_norm=10.0):
    """Update contrastive head with InfoNCE loss."""
    buf_data = instr_buffer.get(device)
    labels = buf_data['instr_label']
    n = labels.size(0)
    if n == 0:
        return 0.0

    buf = agent.buffer
    valid = min(buf.size, n)
    if valid == 0:
        return 0.0

    # Sample states from buffer
    sample_size = min(valid, 128)
    chosen = np.random.choice(valid, sample_size, replace=False)

    states_list = []
    labels_list = []
    for ci in chosen:
        if ci < len(buf.state):
            s = buf.state[ci]  # (grid_num, state_dim)
            g = np.random.randint(0, s.shape[0])
            states_list.append(s[g])
            instr_idx = min(ci, n - 1)
            labels_list.append(labels[instr_idx].item())

    if not states_list:
        return 0.0

    states_t = torch.stack(states_list).to(device)
    labels_t = torch.LongTensor(labels_list).to(device)

    with torch.no_grad():
        state_emb = agent.actor.get_state_emb(states_t)

    loss = contrastive_head.infonce_loss(state_emb.detach(), all_instr_tensor, labels_t)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(contrastive_head.parameters(), max_grad_norm)
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_parameter()

    # Global random seed for reproducibility
    seed = int(os.environ.get("DTMW_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available()
                           and os.environ.get('DTMW_CPU', '0') != '1' else 'cpu')

    env = create_environment(args)
    initialize_logging(args)

    # --- Initialize LLM components ---
    print("[DTMW] Initializing SBERT encoder...")
    sbert = SemanticEncoder(model_name=args.sbert_model, device=str(device))
    sbert_dim = sbert.embedding_dim

    with open(args.scenario_config, 'r') as f:
        scenario_config = json.load(f)

    args.effective_switch_probability = args.switch_probability
    with open(os.path.join(args.log_dir, 'setting.txt'), 'a') as f:
        f.write(f'effective_switch_probability : {args.effective_switch_probability}\n')
    print(f"[DTMW] Effective switch probability: {args.effective_switch_probability}")

    print("[DTMW] Initializing UnifiedLLMPriorManager...")
    llm_config = {
        'model': (os.environ.get('DTMW_MODEL')
                  or os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')),
        'base_url': os.environ.get('OPENAI_BASE_URL'),
        'api_key': os.environ.get('OPENAI_API_KEY'),
    }
    base_manager = UnifiedManager(
        scenarios_path=args.scenario_config,
        sbert=sbert,
        use_llm=args.use_llm,
        llm_config=llm_config,
        update_interval=args.manager_update_interval)
    llm_prior = UnifiedLLMPriorManager(base_manager)

    all_scenario_embs = base_manager.get_all_embeddings()

    # --- DT-MW: Dual feedback controller ---
    dual_controller = DualController(
        xi=args.dual_lr,
        kappa=args.dual_prior_pull,
        lambda_max=args.dual_lambda_max)
    print(f"[DTMW] DualController: xi={args.dual_lr}, "
          f"kappa={args.dual_prior_pull}, lambda_max={args.dual_lambda_max}")

    # Set initial instruction embedding in env
    if not args.no_ssa:
        init_instr = sbert.encode(scenario_config['scenarios'][0]['instruction'])
        env.global_instruction_dim = sbert_dim
        env.global_instruction_vector = init_instr
        env.set_global_instruction_vector(init_instr)
    else:
        env.global_instruction_dim = 0
        env.global_instruction_vector = None
    env.state_space = None  # Force recalculation with llm_dim

    print(f"[DTMW] SBERT dim={sbert_dim}, scenarios={len(scenario_config['scenarios'])}")

    # --- Create standalone PPO agent (MLP, no GNN) ---
    args.memory_size = int(args.TIME_LEN * args.parallel_episode)
    agent = PPOAgent(env, args, device).to(device)

    print(f"[DTMW] Actor params: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"[DTMW] Critic params: {sum(p.numel() for p in agent.critic.parameters()):,}")

    # --- Create contrastive head ---
    contrastive_head = ContrastiveHead(
        state_dim=128,
        instr_dim=sbert_dim,
        proj_dim=64,
        temperature=scenario_config.get('contrastive_temperature', 0.07)
    ).to(device)

    # --- Create reward shaper ---
    reward_shaper = DynamicRewardShaper(gmv_scale=0.01)

    # --- Create instruction buffer ---
    instr_buffer = InstructionBuffer(
        capacity=args.TIME_LEN * args.parallel_episode,
        instr_dim=sbert_dim,
        num_agents=args.grid_num)

    # --- Load multi-day weather data ---
    weather_days = []
    weather_sample_weights = None
    weather_override = os.getenv("DTMW_WEATHER_CONFIG")
    weather_config_path = (Path(weather_override).expanduser() if weather_override
                           else ROOT / "config" / "weather_days.json")
    root = legacy_root() or ROOT
    data_dir = Path(os.getenv("DTMW_DATA_DIR", root / "data")).expanduser()
    is_nyc_data = args.custom_data_path and Path(args.custom_data_path).name.startswith("NYC2015")
    use_weather = (
        not args.synthetic and args.use_real_orders and args.custom_data_path
        and not args.no_manager and (weather_override or is_nyc_data)
        and os.getenv("DTMW_SKIP_WEATHER", os.getenv("DTMW4_SKIP_WEATHER", "0")) != "1"
    )
    if use_weather:
        print("[DTMW] Loading multi-day weather data")
        weather_days, weather_sample_weights = load_weather_days(
            weather_config_path, data_dir)
        print(f"[DTMW] Loaded {len(weather_days)} weather days for training")
    else:
        print("[DTMW] Single-day mode")

    # --- Resume ---
    resume_path = os.environ.get("DTMW_RESUME_PATH") or os.environ.get("COOPRIDE_RESUME_PATH")
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        agent.load_param(resume_path)
        resume_dir = resume_path if os.path.isdir(resume_path) else os.path.dirname(resume_path)
        resume_name = os.path.splitext(os.path.basename(resume_path))[0]
        contrastive_candidates = []
        if resume_name == 'Best_GMV':
            contrastive_candidates.append('contrastive_best_gmv.pt')
        elif resume_name == 'Best_ORR':
            contrastive_candidates.append('contrastive_best_orr.pt')
        contrastive_candidates.extend(['contrastive_best_gmv.pt', 'contrastive_best_orr.pt', 'contrastive_best.pt'])
        for cl_name in contrastive_candidates:
            cl_path = os.path.join(resume_dir, cl_name)
            if os.path.exists(cl_path):
                contrastive_head.load_state_dict(torch.load(cl_path, map_location=device))
                print(f"Loaded contrastive checkpoint: {cl_path}")
                break

    # --- Writer ---
    writer = SummaryWriter(args.log_dir)

    # --- Train ---
    print("[DTMW] Starting training (LLM prior + dual feedback controller)...")
    try:
        train(env, agent, writer, args, device,
              sbert, llm_prior, dual_controller,
              reward_shaper, contrastive_head, instr_buffer,
              all_scenario_embs, scenario_config,
              weather_days=weather_days,
              weather_sample_weights=weather_sample_weights)
    finally:
        writer.close()

    print("[DTMW] Training complete.")


if __name__ == "__main__":
    main()
