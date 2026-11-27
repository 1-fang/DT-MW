"""
DT-MW Evaluation Script (LLM Prior + Dual Feedback)
===================================================
Uses LLM semantic prior plus DualController feedback correction.

Semantic flow:
  context -> UnifiedManager -> instruction + lambda_prior + thresholds
  real ORR/Fairness feedback -> DualController -> lambda -> beta
  -> SSA state augmentation -> Worker inference

Usage:
  export DTMW_RESUME_PATH=/path/to/checkpoint_dir
  export DTMW_EVAL_DATA_PATH=/path/to/data.pkl
  export DTMW_GRID_NUM=289
  export HF_HUB_OFFLINE=1
  python run/eval.py
"""

import os
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import csv
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run.environment import ROOT, configure_environment, create_environment

from llm.unified_manager import UnifiedManager
from llm.fusion_manager import (
    UnifiedLLMPriorManager, DualController, beta_to_lambda, lambda_to_beta,
)
from llm.semantic_encoder import SemanticEncoder
from llm.reward_shaper import DynamicRewardShaper
from llm_algo.mappo_llm import PPOAgent


# ---------------------------------------------------------------------------
# Known weather descriptions for common eval datasets
# When weather_text is not explicitly provided, we look up by filename so the
# Manager still receives meaningful external context rather than empty string.
# ---------------------------------------------------------------------------

_DATASET_WEATHER_MAP = {
    'NYC2015Jan26':  'Severe Blizzard Juno strikes New York City. Historic snowstorm, road closures, state of emergency. Coverage is top priority.',
    'Jan26_blizzard':'Severe Blizzard Juno strikes New York City. Historic snowstorm, road closures, state of emergency. Coverage is top priority.',
    'NYC2015Jan22':  'High-demand surge period spanning Thursday through Saturday. Order volume 3.6x normal. Extreme supply shortage.',
    'NYC2015Jan05':  'Clear cold winter Monday in New York City. Temperature -1°C, light wind. Normal weekday commuter demand.',
    'NYC2015Jan06':  'Cold clear Tuesday in New York City. Temperature -2°C. Normal weekday demand.',
    '_vip':          'High-value VIP event day in New York City. Premium corporate events dominate. Revenue maximization priority.',
}

def infer_weather_text(data_path: str) -> str:
    """Infer weather description from data file path.
    VIP suffix is checked first since its key (_vip) is shorter than date keys
    but must take priority when present.
    """
    if not data_path:
        return ''
    name = Path(data_path).stem
    # VIP is a special override: check before date-based keys
    if '_vip' in name:
        return _DATASET_WEATHER_MAP['_vip']
    # Remaining keys: longer matches win (more specific date strings first)
    for key in sorted(_DATASET_WEATHER_MAP, key=len, reverse=True):
        if key != '_vip' and key in name:
            return _DATASET_WEATHER_MAP[key]
    return ''


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_eval_args():
    args = SimpleNamespace()

    args.device = os.environ.get('DTMW_DEVICE', os.environ.get('COOPRIDE_DEVICE', 'gpu'))
    args.dispatch_interval = 10
    args.speed = args.dispatch_interval
    args.wait_time = args.dispatch_interval
    args.TIME_LEN = int(1440 // args.dispatch_interval)
    configure_environment(args, default_grid=289)
    args.city_time_start = 0
    args.dynamic_env = False

    args.custom_data_path = (os.environ.get('DTMW_EVAL_DATA_PATH') or
                              os.environ.get('DTMW_DATA_PATH') or
                              os.environ.get('COOPRIDE_DATA_PATH'))
    args.use_real_orders = os.environ.get(
        'DTMW_USE_REAL_ORDERS', os.environ.get('COOPRIDE_USE_REAL_ORDERS', '0' if args.synthetic else '1')) == '1'
    args.real_order_sample_rate = float(os.environ.get(
        'DTMW_REAL_ORDER_SAMPLE_RATE',
        os.environ.get('COOPRIDE_REAL_ORDER_SAMPLE_RATE', '0.35')))

    # Weather context (explicit external context, not just env_stats)
    args.eval_weather_text = (os.environ.get('DTMW_EVAL_WEATHER_TEXT') or
                               infer_weather_text(args.custom_data_path))

    args.batch_size = 1000
    args.actor_lr = float(os.environ.get('DTMW_LR', '1e-3'))
    args.critic_lr = float(os.environ.get('DTMW_LR', '1e-3'))
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
    args.parallel_episode = 5
    args.steps_per_epoch = 144
    args.new_order_entropy = True
    args.order_grid = True
    args.feature_normal = int(os.environ.get('DTMW_FEATURE_NORMAL', os.environ.get('COOPRIDE_FEATURE_NORMAL', '3')))

    args.use_llm = os.environ.get('DTMW_USE_LLM', '0') == '1'
    args.scenario_config = os.environ.get(
        'DTMW_SCENARIO_CONFIG',
        ROOT / 'config' / 'scenarios.json')
    args.sbert_model = os.environ.get(
        'DTMW_SBERT_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    args.manager_update_interval = int(os.environ.get('DTMW_MANAGER_INTERVAL', '3'))
    args.drs_weight = float(os.environ.get('DTMW_DRS_WEIGHT', '0.5'))
    args.no_ssa = os.environ.get('DTMW_NO_SSA', '0') == '1'
    args.no_manager = os.environ.get('DTMW_NO_MANAGER', '0') == '1'
    args.memory_size = int(args.TIME_LEN * args.parallel_episode)
    args.base_reward_orr_weight = float(os.environ.get('DTMW_BASE_ORR_WEIGHT', '1.0'))
    args.use_llm_prior = os.environ.get(
        'DTMW_USE_LLM_PRIOR', os.environ.get('DTMW4_USE_LLM_PRIOR', '1')) == '1'
    args.no_dual = os.environ.get(
        'DTMW_NO_DUAL', os.environ.get('DTMW4_NO_DUAL', '0')) == '1'
    args.dual_lr = float(os.environ.get(
        'DTMW_DUAL_LR', os.environ.get('DTMW4_DUAL_LR', '0.10')))
    args.dual_prior_pull = float(os.environ.get(
        'DTMW_DUAL_PRIOR_PULL', os.environ.get('DTMW4_DUAL_PRIOR_PULL', '0.05')))
    args.dual_lambda_max = float(os.environ.get(
        'DTMW_DUAL_LAMBDA_MAX', os.environ.get('DTMW4_DUAL_LAMBDA_MAX', '20.0')))

    args.resume_path = os.environ.get('DTMW_RESUME_PATH') or os.environ.get('COOPRIDE_RESUME_PATH')
    if not args.resume_path:
        raise ValueError('DTMW_RESUME_PATH is required for evaluation.')

    args.eval_repeats = int(os.environ.get('DTMW_EVAL_REPEATS', '1'))
    args.eval_greedy = os.environ.get('DTMW_EVAL_GREEDY', '0') == '1'
    args.eval_tag = os.environ.get('DTMW_EVAL_TAG', 'eval')

    default_output = ROOT / 'result' / 'eval_outputs'
    timestamp = time.strftime('%Y%m%d_%H-%M-%S')
    args.eval_output = os.environ.get(
        'DTMW_EVAL_OUTPUT',
        str(default_output / f'{args.eval_tag}_{timestamp}.csv'))
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_driver_incomes(env):
    return DynamicRewardShaper.collect_driver_incomes(env)


def collect_node_gmvs(env):
    return [float(getattr(node, 'gmv', 0.0)) for node in getattr(env, 'nodes', []) if node is not None]


def get_static_manager_context(scenario_config, sbert):
    """Static fallback used by no-manager / no-LLM-prior ablations."""
    scenarios = scenario_config.get('scenarios', [])
    if scenarios:
        scenario = scenarios[0]
        instr_text = scenario.get('instruction', 'Standard operation.')
        beta = list(scenario.get('beta', [0.4, 0.4, 0.2]))
        label = int(scenario.get('label', 0))
    else:
        instr_text = 'Standard operation.'
        beta = [0.4, 0.4, 0.2]
        label = 0
    instr_emb = sbert.encode(instr_text)
    return instr_text, beta, label, instr_emb


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------

def evaluate_once(env, agent, args, manager, dual_controller, device, run_seed):
    """One evaluation episode. Returns metrics dict."""
    seed_everything(run_seed)
    if args.dynamic_env:
        env.reset_randomseed(run_seed * 1000)
    else:
        env.reset_randomseed(args.env_seed * 1000)

    use_real = getattr(args, 'use_real_orders', False)
    gen_order = 0 if use_real else 1
    weather_text = getattr(args, 'eval_weather_text', '')

    # Initial manager output — use external context from the start
    if args.no_manager or not args.use_llm_prior:
        current_instr_text = args.static_instruction_text
        current_instr_emb = args.static_instruction_emb
        current_lambda_prior = args.static_lambda_prior
        current_rho_min = args.static_rho_min
        current_phi_min = args.static_phi_min
    else:
        init_context = manager.build_context(
            weather_text=weather_text,
            time_slot=0,
            env_stats={})
        mgr_out = manager.step(init_context, 0)
        current_instr_text = mgr_out['instruction_text']
        current_instr_emb  = mgr_out['instruction_emb']
        current_lambda_prior = mgr_out['lambda_prior']
        current_rho_min = mgr_out['rho_min']
        current_phi_min = mgr_out['phi_min']
    dual_controller.reset(current_lambda_prior)
    if args.no_dual:
        current_dual = {
            'lambda': np.asarray(current_lambda_prior, dtype=np.float32),
            'beta': lambda_to_beta(current_lambda_prior),
        }
    else:
        current_dual = dual_controller.step(
            current_lambda_prior, current_rho_min, current_phi_min,
            observed_orr=0.5, observed_fair=0.5)
    current_lambda = current_dual['lambda']
    current_beta = current_dual['beta']

    if not args.no_ssa:
        env.set_global_instruction_vector(current_instr_emb)

    gmv_list = []
    order_response_rates = []
    per_step_rows = []

    states_node, _, order_states, order_idx, _, _ = env.reset(
        mode='PPO2', generate_order=gen_order)
    state = agent.process_state(states_node, 0)
    order, mask_order = agent.process_order(order_states, 0)

    for step_t in range(args.TIME_LEN):
        # Manager periodic update — build full external context each time
        if step_t % args.manager_update_interval == 0:
            total_drivers = sum(n.idle_driver_num for n in env.nodes if n)
            total_orders  = sum(n.real_order_num  for n in env.nodes if n)
            env_stats = {
                'total_drivers':     total_drivers,
                'total_orders':      total_orders,
                'sd_ratio':          total_drivers / max(total_orders, 1),
                'avg_orr':           order_response_rates[-1] if order_response_rates else 0.5,
                'dispatch_interval': args.dispatch_interval,
                'step_t':            step_t,
            }
            if (not args.no_manager) and args.use_llm_prior:
                context = manager.build_context(
                    weather_text=weather_text,
                    time_slot=step_t,
                    env_stats=env_stats,
                )
                mgr_out = manager.step(context, step_t)
                current_instr_text = mgr_out['instruction_text']
                current_instr_emb  = mgr_out['instruction_emb']
                current_lambda_prior = mgr_out['lambda_prior']
                current_rho_min = mgr_out['rho_min']
                current_phi_min = mgr_out['phi_min']
            else:
                current_instr_text = args.static_instruction_text
                current_instr_emb = args.static_instruction_emb
                current_lambda_prior = args.static_lambda_prior
                current_rho_min = args.static_rho_min
                current_phi_min = args.static_phi_min

            incomes_now = collect_driver_incomes(env)
            observed_fair = (1.0 - DynamicRewardShaper._gini(incomes_now)
                             if incomes_now else 0.5)
            if args.no_dual:
                current_dual = {
                    'lambda': np.asarray(current_lambda_prior, dtype=np.float32),
                    'beta': lambda_to_beta(current_lambda_prior),
                }
            else:
                current_dual = dual_controller.step(
                    current_lambda_prior,
                    current_rho_min,
                    current_phi_min,
                    observed_orr=env_stats['avg_orr'],
                    observed_fair=observed_fair)
            current_lambda = current_dual['lambda']
            current_beta = current_dual['beta']
            if not args.no_ssa:
                env.set_global_instruction_vector(current_instr_emb)

        with torch.no_grad():
            _, _, _, _, _, _, _, action_ids, _ = agent.action(
                state, order, mask_order, order_idx,
                sample=not args.eval_greedy)

        orders = env.get_orders_by_id(action_ids)
        next_states_node, next_order_states, next_order_idx, _ = env.step(
            orders, generate_order=gen_order, mode='PPO2')

        gmv_list.append(float(env.gmv))
        if env.order_response_rate >= 0:
            order_response_rates.append(float(env.order_response_rate))

        incomes_detail = collect_driver_incomes(env)
        observed_fair_detail = (1.0 - DynamicRewardShaper._gini(incomes_detail)
                                if incomes_detail else 0.5)
        per_step_rows.append({
            'step':      step_t,
            'gmv':       float(env.gmv),
            'orr':       float(env.order_response_rate),
            'instruction': current_instr_text[:120],
            'beta_gmv':  float(current_beta[0]),
            'beta_orr':  float(current_beta[1]),
            'beta_fair': float(current_beta[2]),
            'lambda_prior_orr': float(current_lambda_prior[0]),
            'lambda_prior_fair': float(current_lambda_prior[1]),
            'lambda_orr': float(current_lambda[0]),
            'lambda_fair': float(current_lambda[1]),
            'rho_min': float(current_rho_min),
            'phi_min': float(current_phi_min),
            'orr_violation': float(current_rho_min - (order_response_rates[-1] if order_response_rates else 0.5)),
            'fair_violation': float(current_phi_min - observed_fair_detail),
        })

        state = agent.process_state(next_states_node, step_t + 1)
        order, mask_order = agent.process_order(next_order_states, step_t + 1)
        order_idx = next_order_idx

    incomes = collect_driver_incomes(env)
    driver_income_gini = DynamicRewardShaper._gini(incomes)
    node_gmv_gini = DynamicRewardShaper._gini(collect_node_gmvs(env))
    return {
        'mean_orr':  float(np.mean(order_response_rates)) if order_response_rates else 0.0,
        'final_orr': float(order_response_rates[-1]) if order_response_rates else 0.0,
        'total_gmv': float(np.sum(gmv_list)),
        'gini':      float(driver_income_gini),
        'driver_income_gini': float(driver_income_gini),
        'node_gmv_gini': float(node_gmv_gini),
        'driver_count': len(incomes),
        'steps':     len(per_step_rows),
        'detail_rows': per_step_rows,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_summary(output_path, rows):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['repeat', 'mean_orr', 'final_orr', 'total_gmv', 'gini', 'steps',
              'driver_income_gini', 'node_gmv_gini', 'driver_count',
              'checkpoint', 'dataset', 'weather_text', 'use_llm', 'eval_greedy']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def write_detail(output_path, rows):
    detail_path = Path(output_path).with_name(Path(output_path).stem + '_detail.csv')
    fields = ['repeat', 'step', 'gmv', 'orr', 'instruction',
              'beta_gmv', 'beta_orr', 'beta_fair',
              'lambda_prior_orr', 'lambda_prior_fair',
              'lambda_orr', 'lambda_fair', 'rho_min', 'phi_min',
              'orr_violation', 'fair_violation']
    with detail_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return detail_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DT-MW. See the README for environment variables.")
    parser.parse_args()
    args = get_eval_args()
    if args.eval_repeats < 1 or not 0 < args.real_order_sample_rate <= 1:
        parser.error("DTMW_EVAL_REPEATS must be positive and DTMW_REAL_ORDER_SAMPLE_RATE must be in (0, 1]")
    base_seed = int(os.environ.get('DTMW_SEED', '42'))
    seed_everything(base_seed)

    device = torch.device(
        'cuda' if args.device == 'gpu' and torch.cuda.is_available()
        and os.environ.get('DTMW_CPU', '0') != '1' else 'cpu')

    print('[DTMW-EVAL] Creating environment...')
    env = create_environment(args)

    print('[DTMW-EVAL] Initializing SBERT encoder...')
    sbert = SemanticEncoder(model_name=args.sbert_model, device=str(device))
    sbert_dim = sbert.embedding_dim

    with open(args.scenario_config, 'r') as f:
        scenario_config = json.load(f)

    # UnifiedManager gives semantic prior; DualController computes final beta.
    print('[DTMW-EVAL] Initializing UnifiedLLMPriorManager + DualController...')
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
    manager = UnifiedLLMPriorManager(base_manager)
    dual_controller = DualController(
        xi=args.dual_lr,
        kappa=args.dual_prior_pull,
        lambda_max=args.dual_lambda_max)
    if not args.use_llm_prior:
        print('[DTMW-EVAL] No-LLM-prior ablation: using static prior.')
    if args.no_dual:
        print('[DTMW-EVAL] No-dual ablation: using prior lambda directly.')

    weather_text = getattr(args, 'eval_weather_text', '')
    print(f'[DTMW-EVAL] Weather context: "{weather_text[:80]}"'
          if weather_text else '[DTMW-EVAL] Weather context: (none, using env_stats fallback)')

    static_instr_text, static_beta, static_label, static_instr_emb = \
        get_static_manager_context(scenario_config, sbert)
    args.static_instruction_text = static_instr_text
    args.static_instruction_emb = static_instr_emb
    args.static_lambda_prior = beta_to_lambda(np.array(static_beta, dtype=np.float32))
    args.static_rho_min = 0.60
    args.static_phi_min = 0.55

    # SSA: inject initial instruction embedding into env
    if not args.no_ssa:
        if args.no_manager or not args.use_llm_prior:
            init_emb = args.static_instruction_emb
        else:
            init_ctx = manager.build_context(weather_text=weather_text, time_slot=0, env_stats={})
            init_out = manager.step(init_ctx, 0)
            init_emb = init_out['instruction_emb']
        env.global_instruction_dim = sbert_dim
        env.global_instruction_vector = init_emb
        env.set_global_instruction_vector(init_emb)
    else:
        env.global_instruction_dim = 0
        env.global_instruction_vector = None
    env.state_space = None

    # Load PPO agent
    print(f'[DTMW-EVAL] Loading checkpoint: {args.resume_path}')
    agent = PPOAgent(env, args, device).to(device)
    agent.load_param(args.resume_path)
    agent.actor.eval()
    agent.critic.eval()

    print(f'[DTMW-EVAL] Mode: {"greedy" if args.eval_greedy else "sample"}, '
          f'repeats={args.eval_repeats}')

    checkpoint_path = str(Path(args.resume_path).resolve())
    dataset_path = str(Path(args.custom_data_path).resolve()) if args.custom_data_path else ''

    summary_rows = []
    all_detail_rows = []

    for repeat_idx in range(args.eval_repeats):
        run_seed = base_seed + repeat_idx
        metrics = evaluate_once(env, agent, args, manager, dual_controller, device, run_seed)
        summary_rows.append({
            'repeat':      repeat_idx + 1,
            'mean_orr':    f"{metrics['mean_orr']:.6f}",
            'final_orr':   f"{metrics['final_orr']:.6f}",
            'total_gmv':   f"{metrics['total_gmv']:.2f}",
            'gini':        f"{metrics['gini']:.6f}",
            'driver_income_gini': f"{metrics['driver_income_gini']:.6f}",
            'node_gmv_gini': f"{metrics['node_gmv_gini']:.6f}",
            'driver_count': metrics['driver_count'],
            'steps':       metrics['steps'],
            'checkpoint':  checkpoint_path,
            'dataset':     dataset_path,
            'weather_text': weather_text[:100],
            'use_llm':     args.use_llm,
            'eval_greedy': args.eval_greedy,
        })
        for row in metrics['detail_rows']:
            all_detail_rows.append({'repeat': repeat_idx + 1, **row})
        print(f'[DTMW-EVAL] Repeat {repeat_idx+1}/{args.eval_repeats} '
              f'ORR={metrics["mean_orr"]:.4f} GMV={metrics["total_gmv"]:.0f} '
              f'DriverGini={metrics["driver_income_gini"]:.4f} '
              f'NodeGMVGini={metrics["node_gmv_gini"]:.4f}')

    summary_path = write_summary(args.eval_output, summary_rows)
    detail_path  = write_detail(args.eval_output, all_detail_rows)

    mean_orr  = np.mean([float(r['mean_orr'])  for r in summary_rows])
    mean_gmv  = np.mean([float(r['total_gmv']) for r in summary_rows])
    mean_gini = np.mean([float(r['driver_income_gini']) for r in summary_rows])
    mean_node_gini = np.mean([float(r['node_gmv_gini']) for r in summary_rows])
    print(f'[DTMW-EVAL] ORR={mean_orr:.4f}  GMV={mean_gmv:.0f}  '
          f'DriverGini={mean_gini:.4f}  NodeGMVGini={mean_node_gini:.4f}')
    print(f'[DTMW-EVAL] Summary → {summary_path}')
    print(f'[DTMW-EVAL] Detail  → {detail_path}')


if __name__ == '__main__':
    main()
