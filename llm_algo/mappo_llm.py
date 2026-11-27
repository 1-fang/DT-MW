"""
Standalone MLP-based MAPPO worker for DT-MW.

No GNN/DGCN dependency. Pure MLP [256, 128] observation encoder.
Includes: Actor, Critic, ContrastiveHead, ReplayBuffer, PPO agent.
"""

import os
import math
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch import float32


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def layer_init(layer, std=1.0, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ValueNorm(nn.Module):
    """Running mean/var normalizer for value targets."""
    def __init__(self, shape, eps=1e-5):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))
        self.register_buffer('count', torch.tensor(eps))
        self.eps = eps

    def update(self, x):
        batch_mean = x.mean(0)
        batch_var = x.var(0)
        batch_count = x.shape[0]
        delta = batch_mean - self.running_mean
        total = self.count + batch_count
        self.running_mean += delta * batch_count / total
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.running_var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.running_mean) / (torch.sqrt(self.running_var) + self.eps)

    def denormalize(self, x):
        return x * (torch.sqrt(self.running_var) + self.eps) + self.running_mean


# ---------------------------------------------------------------------------
# State Encoder: MLP [256, 128] — paper Appendix: "observation encoder (MLP: [256, 128])"
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """MLP-based state representation (no GNN).

    Input: [time_id, grid_id, continuous_features..., (optional) llm_embedding...]
    Output: state_emb of dim `emb_dim` (default 128)
    """
    def __init__(self, grid_dim, time_dim, emb_dim=128, contin_dim=10,
                 llm_dim=0, llm_proj_dim=64):
        super().__init__()
        self.grid_dim = grid_dim
        self.time_dim = time_dim
        self.emb_dim = emb_dim
        self.llm_dim = int(llm_dim) if llm_dim else 0
        self.llm_proj_dim = int(llm_proj_dim) if self.llm_dim else 0

        gain = nn.init.calculate_gain('relu')
        half = emb_dim // 2  # 64

        # Embeddings for discrete features
        self.grid_embedding = layer_init(nn.Embedding(grid_dim, half), std=gain)
        self.time_embedding = layer_init(nn.Embedding(time_dim, half), std=gain)

        # LLM instruction projection (384 -> 64)
        base_contin = contin_dim - self.llm_dim if self.llm_dim else contin_dim
        self.base_contin_dim = max(base_contin, 0)
        if self.llm_dim > 0:
            self.llm_proj = layer_init(nn.Linear(self.llm_dim, self.llm_proj_dim), std=gain)
            contin_in = self.base_contin_dim + self.llm_proj_dim
        else:
            contin_in = contin_dim

        # Continuous feature encoder
        self.contin_fc = layer_init(nn.Linear(contin_in, half), std=gain)

        # Fusion MLP: [time_emb(64) | grid_emb(64) | contin_emb(64)] -> 256 -> 128
        self.fc1 = layer_init(nn.Linear(half * 3, 256), std=gain)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = layer_init(nn.Linear(256, emb_dim), std=1.0)

    def forward(self, state):
        """state: (..., state_dim) where state[...,0]=time, state[...,1]=grid_id, state[...,2:]=contin"""
        time = state[..., 0].long().clamp(0, self.time_dim - 1)
        grid = state[..., 1].long().clamp(0, self.grid_dim - 1)
        contin = state[..., 2:].float()

        time_emb = self.time_embedding(time)
        grid_emb = self.grid_embedding(grid)

        if self.llm_dim > 0:
            base = contin[..., :self.base_contin_dim]
            llm = contin[..., self.base_contin_dim:self.base_contin_dim + self.llm_dim]
            llm_feat = F.relu(self.llm_proj(llm * 10.0))
            if self.base_contin_dim > 0:
                contin_in = torch.cat([base, llm_feat], dim=-1)
            else:
                contin_in = llm_feat
        else:
            contin_in = contin

        contin_emb = self.contin_fc(contin_in)
        x = torch.cat([time_emb, grid_emb, contin_emb], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Order Encoder
# ---------------------------------------------------------------------------

class OrderEncoder(nn.Module):
    """Encode order features into embedding space for pointer-network attention."""
    def __init__(self, grid_dim, time_dim, emb_dim=128, order_contin_dim=4):
        super().__init__()
        self.grid_dim = grid_dim
        gain = nn.init.calculate_gain('relu')
        half = emb_dim // 2
        self.origin_emb = layer_init(nn.Embedding(grid_dim, half), std=gain)
        self.dest_emb = layer_init(nn.Embedding(grid_dim, half), std=gain)
        self.contin_fc = layer_init(nn.Linear(order_contin_dim, half), std=gain)
        self.fc = layer_init(nn.Linear(half * 3, emb_dim), std=1.0)

    def forward(self, order):
        """order: (..., order_dim) where [0]=origin, [1]=dest, [2:]=continuous"""
        origin = order[..., 0].long().clamp(0, self.grid_dim - 1)
        dest = order[..., 1].long().clamp(0, self.grid_dim - 1)
        contin = order[..., 2:].float()
        o_emb = self.origin_emb(origin)
        d_emb = self.dest_emb(dest)
        c_emb = self.contin_fc(contin)
        return self.fc(torch.cat([o_emb, d_emb, c_emb], dim=-1))


# ---------------------------------------------------------------------------
# Actor: Pointer Network (state_emb dot order_emb)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, grid_dim, time_dim, emb_dim, state_contin_dim,
                 order_contin_dim, llm_dim=0, llm_proj_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.state_layer = StateEncoder(
            grid_dim, time_dim, emb_dim, state_contin_dim,
            llm_dim=llm_dim, llm_proj_dim=llm_proj_dim)
        self.order_layer = OrderEncoder(grid_dim, time_dim, emb_dim, order_contin_dim)

    def forward(self, state, order, mask, return_logit=False):
        """
        state: (N, state_dim)
        order: (N, max_order, order_dim)
        mask:  (N, max_order) bool
        """
        if mask.dtype is not torch.bool:
            mask = mask.bool()
        state_emb = self.state_layer(state)          # (N, emb)
        order_emb = self.order_layer(order)           # (N, max_order, emb)
        # Pointer attention: scaled dot product
        compat = torch.bmm(state_emb.unsqueeze(1), order_emb.transpose(-2, -1)).squeeze(1)
        compat = compat / math.sqrt(self.emb_dim)
        compat = compat - compat.max(-1, keepdim=True)[0]
        compat[~mask] = -math.inf
        if return_logit:
            return compat
        return F.softmax(compat, dim=-1)

    def get_state_emb(self, state):
        return self.state_layer(state)


# ---------------------------------------------------------------------------
# Critic: V(s) — single scalar value per grid agent
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    def __init__(self, grid_dim, time_dim, emb_dim, state_contin_dim,
                 llm_dim=0, llm_proj_dim=64):
        super().__init__()
        self.state_layer = StateEncoder(
            grid_dim, time_dim, emb_dim, state_contin_dim,
            llm_dim=llm_dim, llm_proj_dim=llm_proj_dim)
        self.value_head = layer_init(nn.Linear(emb_dim, 1), std=1.0)

    def forward(self, state):
        """Returns V(s) for each grid agent. state: (N, state_dim) -> (N, 1)"""
        emb = self.state_layer(state)
        return self.value_head(F.relu(emb))

    def get_state_emb(self, state):
        return self.state_layer(state)


# ---------------------------------------------------------------------------
# Contrastive Head (unchanged from before)
# ---------------------------------------------------------------------------

class ContrastiveHead(nn.Module):
    def __init__(self, state_dim=128, instr_dim=384, proj_dim=64, temperature=0.07):
        super().__init__()
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.instr_proj = nn.Sequential(
            nn.Linear(instr_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def project_state(self, s):
        return F.normalize(self.state_proj(s), dim=-1)

    def project_instruction(self, e):
        return F.normalize(self.instr_proj(e), dim=-1)

    def intrinsic_reward(self, state_emb, instr_emb):
        s = self.project_state(state_emb)
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0).expand(s.size(0), -1)
        e = self.project_instruction(instr_emb)
        return (s * e).sum(dim=-1)

    def infonce_loss(self, state_embs, all_instr_embs, labels):
        s = self.project_state(state_embs)
        e = self.project_instruction(all_instr_embs)
        logits = s @ e.T / self.temperature
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Instruction Buffer
# ---------------------------------------------------------------------------

class InstructionBuffer:
    def __init__(self, capacity, instr_dim, num_agents):
        self.capacity = capacity
        self.instr_dim = instr_dim
        self.instr_embs = np.zeros((capacity, instr_dim), dtype=np.float32)
        self.labels = np.zeros(capacity, dtype=np.int64)
        self.ptr = 0
        self.size = 0

    def push(self, instr_emb, label):
        self.instr_embs[self.ptr] = instr_emb
        self.labels[self.ptr] = label
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, device='cpu'):
        n = self.size
        return {
            'instr_emb': torch.FloatTensor(self.instr_embs[:n]).to(device),
            'instr_label': torch.LongTensor(self.labels[:n]).to(device),
        }

    def clear(self):
        self.ptr = 0
        self.size = 0


# ---------------------------------------------------------------------------
# Replay Buffer (GAE-based)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Stores transitions for on-policy PPO updates with GAE."""

    def __init__(self, capacity, agent_num, state_dim, max_order_num, order_dim,
                 gamma=0.97, lam=0.95):
        self.capacity = capacity
        self.agent_num = agent_num
        self.gamma = gamma
        self.lam = lam

        # Storage lists (appended per step)
        self.state = []
        self.next_state = []
        self.order = []
        self.action = []
        self.reward = []
        self.value = []
        self.logp = []
        self.mask_order = []
        self.mask_action = []
        self.mask_agent = []
        self.mask_entropy = []

        # GAE results
        self.advantage = []
        self.ret = []

        self.ptr = 0
        self.path_start_idx = 0

    def push(self, state, next_state, order, action, reward, value, logp,
             mask_order, mask_action, mask_agent, mask_entropy):
        self.state.append(state)
        self.next_state.append(next_state)
        self.order.append(order)
        self.action.append(action)
        self.reward.append(reward)
        self.value.append(value)
        self.logp.append(logp)
        self.mask_order.append(mask_order)
        self.mask_action.append(mask_action)
        self.mask_agent.append(mask_agent)
        self.mask_entropy.append(mask_entropy)
        self.ptr += 1

    def finish_path(self, last_value):
        """Compute GAE advantages and returns for the current episode/epoch segment."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.reward[path_slice]
        values = self.value[path_slice]

        n = len(rewards)
        if n == 0:
            self.path_start_idx = self.ptr
            return

        # Append last_value for bootstrapping
        vals = [v for v in values] + [last_value]
        advs = []
        gae = torch.zeros(self.agent_num)
        for t in reversed(range(n)):
            r = rewards[t].squeeze(-1) if rewards[t].dim() > 1 else rewards[t]
            v = vals[t].squeeze(-1) if vals[t].dim() > 1 else vals[t]
            v_next = vals[t + 1].squeeze(-1) if vals[t + 1].dim() > 1 else vals[t + 1]
            delta = r + self.gamma * v_next - v
            gae = delta + self.gamma * self.lam * gae
            advs.insert(0, gae.clone())

        for t_idx, t in enumerate(range(self.path_start_idx, self.ptr)):
            v = values[t_idx].squeeze(-1) if values[t_idx].dim() > 1 else values[t_idx]
            self.advantage.append(advs[t_idx])
            self.ret.append(advs[t_idx] + v)

        self.path_start_idx = self.ptr

    def get(self, device):
        """Return all data as tensors on device."""
        data = {
            'state': torch.stack(self.state).to(device),
            'next_state': torch.stack(self.next_state).to(device),
            'order': torch.stack(self.order).to(device),
            'action': torch.stack(self.action).to(device),
            'reward': torch.stack(self.reward).to(device),
            'value': torch.stack(self.value).to(device),
            'logp': torch.stack(self.logp).to(device),
            'mask_order': torch.stack(self.mask_order).to(device),
            'mask_action': torch.stack(self.mask_action).to(device),
            'mask_agent': torch.stack(self.mask_agent).to(device),
            'mask_entropy': torch.stack(self.mask_entropy).to(device),
            'advantage': torch.stack(self.advantage).to(device),
            'ret': torch.stack(self.ret).to(device),
        }
        return data

    def clear(self):
        self.state.clear()
        self.next_state.clear()
        self.order.clear()
        self.action.clear()
        self.reward.clear()
        self.value.clear()
        self.logp.clear()
        self.mask_order.clear()
        self.mask_action.clear()
        self.mask_agent.clear()
        self.mask_entropy.clear()
        self.advantage.clear()
        self.ret.clear()
        self.ptr = 0
        self.path_start_idx = 0

    @property
    def size(self):
        return self.ptr


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """Standalone MAPPO agent with MLP Actor/Critic for DT-MW."""

    def __init__(self, env, args, device):
        self.args = args
        self.device = device
        self.env = env
        self.agent_num = args.grid_num
        self.TIME_LEN = args.TIME_LEN
        self.emb_dim = 128

        # State / order dims
        self.state_dim = env.get_state_space_node() + 1  # +1 for time prepended
        self.order_dim = 6
        if getattr(args, 'new_order_entropy', True):
            self.order_dim += 2
        self.order_contin_dim = self.order_dim - 2  # minus origin/dest grid ids

        # Max orders per grid
        order_dict = {143: 100, 121: 100, 100: 60, 289: 100}
        self.max_order_num = order_dict.get(args.grid_num, 100)
        max_order_env = os.environ.get("DTMW_MAX_ORDER_NUM") or os.environ.get("COOPRIDE_MAX_ORDER_NUM")
        if max_order_env:
            try:
                self.max_order_num = int(max_order_env)
            except ValueError:
                pass

        # LLM dim
        self.llm_dim = int(getattr(env, "global_instruction_dim", 0) or 0)
        self.llm_proj_dim = 64

        # Feature normalization
        self.feature_normal = getattr(args, 'feature_normal', 3)
        if self.feature_normal == 3:
            root = os.environ.get("DTMW_SIMULATOR_ROOT")
            default_root = Path(root).expanduser() if root else Path(__file__).resolve().parents[1]
            feature_dir = Path(os.getenv("DTMW_FEATURE_DIR", default_root / "assets")).expanduser()
            scope_path = os.path.join(feature_dir, f'feature_{args.grid_num}.pkl')
            if os.path.exists(scope_path):
                with open(scope_path, 'rb') as f:
                    self.feature_scope = pickle.load(f)
            else:
                self.feature_normal = 1
                self.feature_scope = None
        else:
            self.feature_scope = None

        # Hyperparams
        self.gamma = args.gamma
        self.lam = args.lam
        self.clip_ratio = args.clip_ratio
        self.ent_factor = args.ent_factor
        self.max_grad_norm = args.max_grad_norm
        self.batch_size = args.batch_size
        self.minibatch_num = getattr(args, 'minibatch_num', 5)
        self.train_actor_iters = getattr(args, 'train_actor_iters', 1)
        self.train_critic_iters = getattr(args, 'train_critic_iters', 1)
        self.reward_scale = getattr(args, 'reward_scale', 5)
        self.adv_normal = getattr(args, 'adv_normal', True)

        # Networks
        state_contin_dim = self.state_dim - 2  # minus time and grid_id
        self.actor = Actor(
            self.agent_num, self.TIME_LEN, self.emb_dim, state_contin_dim,
            self.order_contin_dim, llm_dim=self.llm_dim, llm_proj_dim=self.llm_proj_dim)
        self.critic = Critic(
            self.agent_num, self.TIME_LEN, self.emb_dim, state_contin_dim,
            llm_dim=self.llm_dim, llm_proj_dim=self.llm_proj_dim)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        # Value normalizer
        self.value_normalizer = ValueNorm(1)

        # Buffer
        self.buffer = ReplayBuffer(
            capacity=args.memory_size, agent_num=self.agent_num,
            state_dim=self.state_dim, max_order_num=self.max_order_num,
            order_dim=self.order_dim, gamma=self.gamma, lam=self.lam)

        self.step_count = 0

    def to(self, device):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        self.value_normalizer.to(device)
        return self

    # ---- State / Order Processing ----

    def process_state(self, s, t):
        """Convert raw node observations to tensor. s: list of arrays."""
        s = np.stack(s, axis=0)
        if self.feature_normal == 3 and self.feature_scope is not None:
            # feature_scope['state'] shape: (2, D) — row 0=min, row 1=max
            state_scope = self.feature_scope['state'].numpy()
            if state_scope.ndim == 2:
                state_scope = state_scope[1]  # use max row
            feature_dim = s.shape[1] - 1
            scope_dim = state_scope.shape[0] - 1
            if feature_dim > scope_dim:
                pad = np.ones((feature_dim - scope_dim,), dtype=state_scope.dtype)
                state_scope = np.concatenate([state_scope, pad], axis=0)
            denom = state_scope[None, 1:1 + feature_dim]
            denom[denom == 0] = 1
            s[:, 1:] /= denom
        elif self.feature_normal == 1:
            feature_max = np.max(np.abs(s[:, 1:]), axis=0, keepdims=True)
            feature_max[feature_max == 0] = 1
            s[:, 1:] /= feature_max

        time_col = np.zeros((s.shape[0], 1), dtype=float)
        time_col[:, 0] = t
        state = np.concatenate([time_col, s], axis=1)
        return torch.Tensor(state)

    def process_order(self, order_state, t):
        """Convert raw order observations to padded tensor + mask."""
        order_num = []
        trimmed = []
        for i in range(len(order_state)):
            cur = order_state[i]
            if len(cur) > self.max_order_num:
                cur = cur[:self.max_order_num]
            trimmed.append(cur)
            order_num.append(len(cur))

        order = torch.zeros((self.agent_num, self.max_order_num, 6), dtype=float32)
        mask = torch.zeros((self.agent_num, self.max_order_num), dtype=torch.bool)
        for i in range(len(trimmed)):
            n = order_num[i]
            if n > 0:
                order[i, :n, :6] = torch.Tensor(trimmed[i])[:, :6]
                mask[i, :n] = True

        # Add entropy features for order-level state summaries
        if getattr(self.args, 'new_order_entropy', True):
            driver_num = torch.Tensor([n.idle_driver_num for n in self.env.nodes]) + 1e-5
            order_cnt = torch.Tensor([n.real_order_num for n in self.env.nodes]) + 1e-5
            do = torch.stack([driver_num, order_cnt], dim=1)
            orr_ent = torch.min(do, dim=1)[0] / torch.max(do, dim=1)[0]
            node_ids = order[:, :, :2].long()
            ent_feat = orr_ent[node_ids[:, :, 1]] - orr_ent[node_ids[:, :, 0]]
            drv_feat = driver_num[node_ids[:, :, 1]] - driver_num[node_ids[:, :, 0]]
            ord_feat = order_cnt[node_ids[:, :, 1]] - order_cnt[node_ids[:, :, 0]]
            order[:, :, 5] = ent_feat
            order = torch.cat([order, drv_feat.unsqueeze(-1), ord_feat.unsqueeze(-1)], dim=-1)

        # Feature normalization
        if self.feature_normal == 3 and self.feature_scope is not None:
            order_scope = self.feature_scope['order'].numpy()
            if order_scope.ndim == 2:
                order_scope = order_scope[1]  # use max row
            od = order[:, :, 2:]
            feat_dim = od.shape[-1]
            scope_dim = order_scope.shape[0] - 2
            if feat_dim > scope_dim:
                pad = np.ones((feat_dim - scope_dim,), dtype=order_scope.dtype)
                order_scope = np.concatenate([order_scope, pad], axis=0)
            denom = torch.from_numpy(order_scope[2:2 + feat_dim]).float()
            denom[denom == 0] = 1
            order[:, :, 2:] = od / denom[None, None, :]
        elif self.feature_normal == 1:
            fmax = torch.max(torch.abs(order[:, :, 2:]))
            if fmax > 0:
                order[:, :, 2:] /= fmax

        return order, mask

    def remove_order_grid(self, order):
        """Keep order grid ids (needed for pointer attention)."""
        return order

    def mask_fake(self, order, mask):
        return mask

    # ---- Action Selection ----

    def action(self, state, order, mask, order_idx, sample=True):
        """Select actions for all grids using pointer-network actor.

        Returns: action, value, logp, mask_agent, mask_order_multi,
                 mask_action, mask_entropy, action_ids, selected_ids
        """
        mask = mask.bool()
        state_d = state.to(self.device)
        order_d = order.to(self.device)
        mask_d = mask.to(self.device)

        with torch.no_grad():
            logits = self.actor(state_d, order_d, mask_d, return_logit=True).cpu()
            values = self.critic(state_d).cpu().squeeze(-1)  # (N,)

        action = torch.zeros((self.agent_num, self.max_order_num), dtype=float32)
        mask_order_multi = torch.zeros(
            (mask.shape[0], mask.shape[1], mask.shape[1]), dtype=torch.bool)
        mask_action = torch.zeros((self.agent_num, self.max_order_num), dtype=torch.bool)
        mask_entropy = torch.zeros((self.agent_num, self.max_order_num), dtype=torch.bool)
        oldp = torch.zeros((self.agent_num, self.max_order_num), dtype=float32)
        mask_agent = torch.ones((self.agent_num,), dtype=torch.bool)
        action_ids = []
        selected_ids = []

        for i in range(self.agent_num):
            driver_num = min(self.env.nodes[i].idle_driver_num, self.max_order_num)
            fake_num = self.env.nodes[i].fleet_order_num + 1
            choose = []

            if driver_num == 0:
                choose = [0]
                mask_agent[i] = 0
            else:
                mask_d_i = mask[i].clone()
                mask_entropy[i, 0] = 1
                logit_full = logits[i].clone()
                logit_full[~mask_d_i] = -math.inf

                for d in range(driver_num):
                    mask_order_multi[i, d] = mask_d_i
                    prob = F.softmax(logit_full, dim=-1)
                    if torch.isnan(prob).any() or torch.isinf(prob).any():
                        chosen_idx = int(torch.argmax(logit_full).item())
                    elif sample:
                        chosen_idx = int(torch.multinomial(prob, 1, replacement=True).item())
                    else:
                        chosen_idx = int(torch.argmax(prob).item())
                    choose.append(chosen_idx)
                    mask_action[i, d] = 1
                    oldp[i, d] = prob[chosen_idx]
                    if chosen_idx >= fake_num:
                        mask_d_i[chosen_idx] = 0
                        logit_full[chosen_idx] = -math.inf

            action[i, :len(choose)] = torch.Tensor(choose)
            action_ids.append([order_idx[i][idx] for idx in choose])
            selected_ids.append(choose if driver_num > 0 else [])

        return (action, values, oldp, mask_agent, mask_order_multi,
                mask_action, mask_entropy, action_ids, selected_ids)

    # ---- PPO Update ----

    def update(self, device=None, writer=None):
        """Run PPO update on collected buffer data."""
        device = device or self.device
        data = self.buffer.get(device)

        # Normalize advantages
        adv = data['advantage']
        if self.adv_normal and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = data['state'].shape[0]
        batch_size = min(T, max(1, self.batch_size // self.agent_num))
        batch_num = max(1, int(np.round(T / batch_size / self.minibatch_num)))

        # Actor update
        for _ in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()
            cnt = 0
            for index in BatchSampler(SubsetRandomSampler(range(T)), batch_size, True):
                loss_a = self._actor_loss(data, adv, index)
                loss_a = loss_a / batch_num
                loss_a.backward()
                if (cnt + 1) % batch_num == 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    self.actor_optimizer.zero_grad()
                cnt += 1
            if cnt % batch_num != 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()

        # Critic update
        for _ in range(self.train_critic_iters):
            self.critic_optimizer.zero_grad()
            cnt = 0
            for index in BatchSampler(SubsetRandomSampler(range(T)), batch_size, True):
                loss_c = self._critic_loss(data, index)
                loss_c = loss_c / batch_num
                loss_c.backward()
                if (cnt + 1) % batch_num == 0:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad()
                cnt += 1
            if cnt % batch_num != 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

        self.step_count += 1
        self.buffer.clear()

    def _actor_loss(self, data, adv, index):
        """Compute clipped PPO actor loss for a minibatch."""
        states = data['state'][index]       # (B, N, state_dim)
        orders = data['order'][index]       # (B, N, max_order, order_dim)
        actions = data['action'][index]     # (B, N, max_order)
        old_logp = data['logp'][index]      # (B, N, max_order)
        mask_action = data['mask_action'][index]  # (B, N, max_order)
        mask_agent = data['mask_agent'][index]    # (B, N)
        mask_order = data['mask_order'][index]    # (B, N, max_order, max_order)
        advantages = adv[index]             # (B, N)

        B, N = states.shape[0], states.shape[1]
        driver_slots = actions.shape[-1]

        # Flatten batch and agent dims for actor forward
        states_flat = states.reshape(B * N, -1)
        orders_flat = orders.reshape(B * N, orders.shape[-2], orders.shape[-1])
        # Reuse the shared compatibility logits and apply per-driver masks afterwards.
        mask_flat = mask_order[:, :, 0, :].reshape(B * N, -1)

        logits = self.actor(states_flat, orders_flat, mask_flat, return_logit=True)
        logits = logits.reshape(B, N, 1, -1).expand(-1, -1, driver_slots, -1)
        masked_logits = logits.masked_fill(~mask_order, -math.inf)
        valid_slots = mask_action & mask_agent.unsqueeze(-1)
        safe_logits = masked_logits.masked_fill(~valid_slots.unsqueeze(-1), 0.0)
        probs = F.softmax(safe_logits, dim=-1).clamp(min=1e-8)

        act_idx = actions.long().unsqueeze(-1)
        new_p = probs.gather(-1, act_idx).squeeze(-1)
        old_p = old_logp.clamp(min=1e-8)

        advantages = advantages.unsqueeze(-1).expand_as(new_p)
        ratio = new_p / old_p
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        loss_pi = -torch.min(ratio * advantages, clip_adv)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1)
        loss = (loss_pi - self.ent_factor * entropy) * valid_slots.float()

        denom = valid_slots.float().sum().clamp(min=1.0)
        return loss.sum() / denom

    def _critic_loss(self, data, index):
        """Compute value function loss for a minibatch."""
        states = data['state'][index]       # (B, N, state_dim)
        returns = data['ret'][index]        # (B, N)
        old_values = data['value'][index]   # (B, N)

        B, N = states.shape[0], states.shape[1]
        states_flat = states.reshape(B * N, -1)
        new_values = self.critic(states_flat).squeeze(-1).reshape(B, N)

        # Clipped value loss
        v_clipped = old_values + torch.clamp(
            new_values - old_values, -self.clip_ratio, self.clip_ratio)
        loss1 = (new_values - returns) ** 2
        loss2 = (v_clipped - returns) ** 2
        return 0.5 * torch.max(loss1, loss2).mean()

    # ---- Save / Load ----

    def save_param(self, save_dir, save_name='param'):
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
        }
        path = os.path.join(save_dir, f'{save_name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_param(self, load_path):
        if os.path.isdir(load_path):
            load_path = os.path.join(load_path, 'param.pkl')
        with open(load_path, 'rb') as f:
            d = pickle.load(f)
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])
        if 'actor_opt' in d:
            self.actor_optimizer.load_state_dict(d['actor_opt'])
        if 'critic_opt' in d:
            self.critic_optimizer.load_state_dict(d['critic_opt'])
