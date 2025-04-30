import math, random, os, time
from collections import deque, namedtuple
from typing import Tuple, Dict, List, Optional
import numpy as np
import gym, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True
from torch.cuda.amp import autocast, GradScaler


Transition = namedtuple('Transition', 's a r ns done w idx')

# ----------------------------- utils ---------------------------------------
def orthogonal_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ----------------------------- env -----------------------------------------
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env); self._skip = skip
    def step(self, action):
        tot_r, done, info = 0.0, False, {}
        for _ in range(self._skip):
            obs, r, done, info = self.env.step(action)
            tot_r += r
            if done: break
        return obs, tot_r, done, info

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, 4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, 4)
    return env

class SumTree:
    def __init__(self, cap):
        self.cap = cap; self.size = 0; self.ptr = 0
        self.tree = np.zeros(2 * cap, dtype=np.float64); self.data = [None] * cap
    def add(self, p, data):
        idx = self.ptr + self.cap; self.data[self.ptr] = data; self.update(idx, p)
        self.ptr = (self.ptr + 1) % self.cap; self.size = min(self.size + 1, self.cap)
    def update(self, idx, p):
        p = np.float64(p)
        change = p - self.tree[idx]; self.tree[idx] = p
        parent = idx // 2
        while parent > 0:
            self.tree[parent] += change; parent //= 2
    def _retrieve(self, idx, s):
        left = 2 * idx
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left] + 1e-5:
            return self._retrieve(left, s)
        else:
            if right < len(self.tree):
                 return self._retrieve(right, s - self.tree[left])
            else:
                 return left
    def sample(self, s):
        idx = self._retrieve(1, s); data_idx = idx - self.cap
        if data_idx < 0 or data_idx >= self.cap or self.data[data_idx] is None:
             return idx, None, 0
        return idx, self.data[data_idx], self.tree[idx]
    @property
    def total(self):
        return self.tree[1]

class PERBuffer:
    def __init__(self, cap=200_000, alpha=0.6):
        self.tree = SumTree(cap); self.alpha = alpha; self.eps = 1e-4
    def add(self, *args):
        max_p = self.tree.tree[self.tree.cap : self.tree.cap + self.tree.size].max() if self.tree.size > 0 else 0.0
        p = max_p if max_p > 0 else 1.0
        self.tree.add(p, args)
    def sample(self, k, beta):
        if self.tree.total == 0: return None
        seg = self.tree.total / k
        batch, idxs, prs = [], [], []
        for i in range(k):
            s = random.uniform(seg * i, seg * (i + 1))
            s = min(s, self.tree.total - 1e-6)
            idx, data, p = self.tree.sample(s)
            if data is None: continue
            batch.append(data); idxs.append(idx); prs.append(p)
        if not batch: return None
        prs = np.array(prs, dtype=np.float64)
        probs = prs / self.tree.total if self.tree.total > 0 else np.zeros_like(prs)
        w = (self.tree.size * probs + 1e-10)**(-beta)
        w = np.nan_to_num(w, nan=0.0, posinf=1.0)
        w /= (w.max() + 1e-10)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return Transition(states, actions, rewards, next_states, dones, w, idxs)
    def update(self, idxs, td):
        priorities = (np.abs(td) + self.eps)**self.alpha
        priorities = priorities.astype(np.float64)
        for i, p in zip(idxs, priorities):
            self.tree.update(i, p)
    def __len__(self): return self.tree.size

class DuelingQNet(nn.Module):
    def __init__(self, c, act):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(c,32,8,4), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,4,2), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1), nn.ReLU(inplace=True), nn.Flatten()
        )
        fd = 64*7*7
        self.value = nn.Sequential(nn.Linear(fd,512), nn.ReLU(inplace=True), nn.Linear(512,1))
        self.adv   = nn.Sequential(nn.Linear(fd,512), nn.ReLU(inplace=True), nn.Linear(512,act))
        self.apply(orthogonal_init_)
        nn.init.orthogonal_(self.value[-1].weight, gain=0.01)
        nn.init.constant_(self.value[-1].bias, 0)
        nn.init.orthogonal_(self.adv[-1].weight, gain=0.01)
        nn.init.constant_(self.adv[-1].bias, 0)
    def forward(self,x):
        x = x.float()/255.0; f = self.feat(x); v = self.value(f); a = self.adv(f)
        return v + a - a.mean(1,keepdim=True)

class ICM(nn.Module):
    def __init__(self, c, act, fd=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(c,32,8,4), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,4,2), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1), nn.ReLU(inplace=True), nn.Flatten(),
            nn.Linear(64*7*7,fd), nn.ReLU(inplace=True)
        )
        self.inv = nn.Sequential(nn.Linear(fd*2,256), nn.ReLU(inplace=True), nn.Linear(256,act))
        self.fwd = nn.Sequential(nn.Linear(fd+act,256), nn.ReLU(inplace=True), nn.Linear(256,fd))
        self.apply(orthogonal_init_)
    def forward(self,s,ns,a1h):
        φs = self.enc(s.float() / 255.0)
        φns = self.enc(ns.float() / 255.0)
        a_pred = self.inv(torch.cat([φs,φns],1))
        φns_hat = self.fwd(torch.cat([φs,a1h],1))
        return φs, φns, a_pred, φns_hat

class Agent:
    def __init__(self, action_space):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.action_space = action_space
        self.q = DuelingQNet(4,action_space).to(self.device)
        self.tgt = DuelingQNet(4,action_space).to(self.device); self.tgt.load_state_dict(self.q.state_dict()); self.tgt.eval()
        self.icm = ICM(4,action_space).to(self.device)

        self.opt_q = optim.Adam(self.q.parameters(), lr=1e-4, eps=1e-4)
        self.opt_icm = optim.Adam(self.icm.parameters(), lr=1e-4, eps=1e-4)

        self.buf = PERBuffer(); self.batch=128; self.gamma=0.99; self.tau=0.50
        self.beta0, self.beta_frames = 0.4, 1_000_000

        self.eps_start = 1.0
        self.eps_end = 0.02
        self.eps_decay_frames = 1_000_000

        self.eta=0.01; self.beta_icm=0.2
        self.alpha_m=0.9; self.tau_m=0.03;
        self.scaler = GradScaler()

        self.train_frequency = 4
        self.target_update_interval = 8000
        self.initial_collect_frames = 0

    @torch.no_grad()
    def act(self, s, f_idx):
        eps_threshold = self.get_epsilon(f_idx)
        if random.random() < eps_threshold:
             return random.randrange(self.action_space)
        else:
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, device=self.device, dtype=torch.uint8)
            elif s.device != self.device:
                s = s.to(self.device)
            if s.ndim == 3: s = s.unsqueeze(0)
            q_values = self.q(s)
            return int(q_values.argmax(1).item())

    def get_epsilon(self, f_idx):
        effective_frames = max(0, f_idx - self.initial_collect_frames)
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * effective_frames / self.eps_decay_frames)
        return max(self.eps_end, eps)

    def get_beta(self, f_idx):
         return min(1.0, self.beta0 + (1 - self.beta0) * max(0, f_idx - self.initial_collect_frames) / self.beta_frames)

    def update_target_network(self):
        for t_param, q_param in zip(self.tgt.parameters(), self.q.parameters()):
            t_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * t_param.data)

    def train_step(self, f_idx) -> Optional[Dict[str, float]]:
        if len(self.buf) < self.batch:
             return None

        beta = self.get_beta(f_idx)
        trans = self.buf.sample(self.batch, beta)

        if trans is None:
            return None

        s = torch.tensor(trans.s, dtype=torch.uint8, device=self.device)
        ns= torch.tensor(trans.ns, dtype=torch.uint8, device=self.device)
        a = torch.tensor(trans.a, dtype=torch.long, device=self.device).unsqueeze(1)
        r_ext = torch.tensor(np.sign(trans.r), dtype=torch.float32, device=self.device).unsqueeze(1)
        done = torch.tensor(trans.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        w = torch.tensor(trans.w, dtype=torch.float32, device=self.device).unsqueeze(1)

        a1h = F.one_hot(a.squeeze(1), num_classes=self.action_space).float()
        φs, φns, a_pred, φns_hat = self.icm(s, ns, a1h)
        inv_loss = F.cross_entropy(a_pred, a.squeeze(1))
        fwd_loss = 0.5 * (φns_hat - φns.detach()).pow(2).mean()
        with torch.no_grad():
             r_int = self.eta * 0.5 * (φns_hat - φns).pow(2).mean(dim=1, keepdim=True)
             r_int = torch.clamp(r_int, 0, 1)

        with torch.no_grad():
            log_pi_s = F.log_softmax(self.q(s) / self.tau_m, dim=1)
            log_pi_sa = log_pi_s.gather(1, a).clamp(-1, 0)
        r_bar = r_ext + r_int * 1000 + self.alpha_m * log_pi_sa

        with torch.no_grad():
            q_ns_current = self.q(ns)
            log_pi_ns = F.log_softmax(q_ns_current / self.tau_m, dim=1)
            pi_ns = log_pi_ns.exp()
            q_ns_target = self.tgt(ns)
            v_ns = (pi_ns * (q_ns_target - self.tau_m * log_pi_ns)).sum(dim=1, keepdim=True)
            y = r_bar + self.gamma * (1 - done) * v_ns

        q_sa = self.q(s).gather(1, a)
        td_error = y - q_sa

        with autocast():
            q_loss = (w * F.smooth_l1_loss(q_sa, y.detach(), reduction='none')).mean()
            icm_loss = (1 - self.beta_icm) * inv_loss + self.beta_icm * fwd_loss
            total_loss = q_loss + icm_loss

        self.opt_q.zero_grad(); self.opt_icm.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.opt_q); self.scaler.unscale_(self.opt_icm)
        nn.utils.clip_grad_norm_(self.q.parameters(), 10)
        nn.utils.clip_grad_norm_(self.icm.parameters(), 10)
        self.scaler.step(self.opt_q); self.scaler.step(self.opt_icm)
        self.scaler.update()

        self.buf.update(trans.idx, td_error.detach().squeeze(1).cpu().numpy())

        log_data = {
            'q_loss': q_loss.item(),
            'icm_loss': icm_loss.item(),
            'inv_loss': inv_loss.item(),
            'fwd_loss': fwd_loss.item(),
            'total_loss': total_loss.item(),
            'q_sa_mean': q_sa.mean().item(),
            'r_ext_mean': r_ext.mean().item(),
            'r_int_mean': r_int.mean().item(),
            'beta': beta
        }
        return log_data

def save_plots(log_dir: str, metrics: Dict[str, List], f_indices_episode: List[int], f_indices_loss: List[int]):
    """Saves plots of training metrics."""
    os.makedirs(log_dir, exist_ok=True)

    if f_indices_episode:
        fig_ep, axs_ep = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        axs_ep[0].plot(f_indices_episode, metrics['episode_rewards'], label='Episode Reward')
        axs_ep[0].plot(f_indices_episode, metrics['mean_rewards'], label='Mean 20 Reward', linestyle='--')
        axs_ep[0].set_ylabel('Reward')
        axs_ep[0].legend()
        axs_ep[0].grid(True)
        axs_ep[0].set_title('Episode Rewards')

        axs_ep[1].plot(f_indices_episode, metrics['episode_steps'], label='Episode Steps')
        axs_ep[1].set_ylabel('Steps')
        axs_ep[1].legend()
        axs_ep[1].grid(True)

        axs_ep[2].plot(f_indices_episode, metrics['epsilons'], label='Epsilon')
        axs_ep[2].set_xlabel('Frames')
        axs_ep[2].set_ylabel('Epsilon')
        axs_ep[2].legend()
        axs_ep[2].grid(True)

        plt.tight_layout()
        fig_ep.savefig(os.path.join(log_dir, "episode_metrics.png"))
        plt.close(fig_ep)

    if f_indices_loss:
        fig_loss, axs_loss = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        axs_loss[0].plot(f_indices_loss, metrics['q_losses'], label='Q Loss', alpha=0.7)
        axs_loss[0].plot(f_indices_loss, metrics['icm_losses'], label='ICM Loss', alpha=0.7)
        axs_loss[0].plot(f_indices_loss, metrics['total_losses'], label='Total Loss', linestyle='--')
        axs_loss[0].set_ylabel('Loss')
        axs_loss[0].legend()
        axs_loss[0].grid(True)
        axs_loss[0].set_title('Network Losses')

        axs_loss[1].plot(f_indices_loss, metrics['inv_losses'], label='Inverse Loss', alpha=0.7)
        axs_loss[1].plot(f_indices_loss, metrics['fwd_losses'], label='Forward Loss', alpha=0.7)
        axs_loss[1].set_ylabel('ICM Component Loss')
        axs_loss[1].legend()
        axs_loss[1].grid(True)

        axs_loss[2].plot(f_indices_loss, metrics['q_sa_means'], label='Avg Q Value (Sampled)')
        axs_loss[2].set_xlabel('Frames')
        axs_loss[2].set_ylabel('Value')
        axs_loss[2].legend()
        axs_loss[2].grid(True)

        plt.tight_layout()
        fig_loss.savefig(os.path.join(log_dir, "loss_metrics.png"))
        plt.close(fig_loss)

    if f_indices_loss:
        fig_rew, axs_rew = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        axs_rew[0].plot(f_indices_loss, metrics['r_ext_means'], label='Avg Extrinsic Reward (Sampled)', alpha=0.7)
        axs_rew[0].plot(f_indices_loss, metrics['r_int_means'], label='Avg Intrinsic Reward (Sampled)', alpha=0.7)
        axs_rew[0].set_ylabel('Reward Value')
        axs_rew[0].legend()
        axs_rew[0].grid(True)
        axs_rew[0].set_title('Sampled Reward Components')

        axs_rew[1].plot(f_indices_loss, metrics['betas'], label='PER Beta')
        axs_rew[1].set_xlabel('Frames')
        axs_rew[1].set_ylabel('Beta Value')
        axs_rew[1].legend()
        axs_rew[1].grid(True)

        plt.tight_layout()
        fig_rew.savefig(os.path.join(log_dir, "reward_and_beta_metrics.png"))
        plt.close(fig_rew)


def train(episodes=10_000, max_steps=30_000, save_interval=10, plot_interval=10,
          log_dir_base="runs/mario_munchausen_plots"):
    env=make_env(); agent=Agent(env.action_space.n)

    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"{log_dir_base}_{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs and plots will be saved to: {log_dir}")

    metrics = {
        'q_losses': [], 'icm_losses': [], 'inv_losses': [], 'fwd_losses': [],
        'total_losses': [], 'q_sa_means': [], 'r_ext_means': [], 'r_int_means': [],
        'betas': [],
        'episode_rewards': [], 'mean_rewards': [], 'episode_steps': [], 'epsilons': []
    }
    f_indices_loss = [] # Frame indices for loss logging steps
    f_indices_episode = [] # Frame indices for episode end steps

    scores=deque(maxlen=20)
    f_idx = 0

    print(f"Collecting initial {agent.initial_collect_frames} experiences...")
    o = env.reset(); o=np.asarray(o); o=np.squeeze(o, axis=-1)
    for frame in trange(agent.initial_collect_frames):
        a = env.action_space.sample()
        no, r, d, info = env.step(a); no=np.asarray(no); no=np.squeeze(no, axis=-1)
        agent.buf.add(o, a, r, no, d)
        o = no
        if d:
            o = env.reset(); o=np.asarray(o); o=np.squeeze(o, axis=-1)
    f_idx = agent.initial_collect_frames
    print("Initial collection complete. Starting training...")

    for ep in trange(episodes):
        o, d = env.reset(), False; o = np.asarray(o); o = np.squeeze(o, axis=-1)
        ep_r, steps = 0.0, 0
        ep_start_time = time.time()

        while not d and steps < max_steps:
            a = agent.act(o, f_idx)
            no, r, d, info = env.step(a); no = np.asarray(no); no = np.squeeze(no, axis=-1)
            agent.buf.add(o, a, r, no, d)
            o = no
            ep_r += r
            steps += 1
            f_idx += 1

            if f_idx > agent.initial_collect_frames and len(agent.buf) >= agent.batch and f_idx % agent.train_frequency == 0:
                log_data = agent.train_step(f_idx)

                if log_data and f_idx % 100 == 0: # Log every 100 training steps
                    f_indices_loss.append(f_idx)
                    metrics['q_losses'].append(log_data['q_loss'])
                    metrics['icm_losses'].append(log_data['icm_loss'])
                    metrics['inv_losses'].append(log_data['inv_loss'])
                    metrics['fwd_losses'].append(log_data['fwd_loss'])
                    metrics['total_losses'].append(log_data['total_loss'])
                    metrics['q_sa_means'].append(log_data['q_sa_mean'])
                    metrics['r_ext_means'].append(log_data['r_ext_mean'])
                    metrics['r_int_means'].append(log_data['r_int_mean'])
                    metrics['betas'].append(log_data['beta'])

            if f_idx > agent.initial_collect_frames and f_idx % agent.target_update_interval == 0:
                agent.update_target_network()

        scores.append(ep_r)
        mean_score = np.mean(scores) if scores else 0.0
        ep_duration = time.time() - ep_start_time
        fps = steps / ep_duration if ep_duration > 0 else 0

        print(f"Ep {ep:<4d} | Frames: {f_idx:<8d} | Steps: {steps:<5d} | R: {ep_r:<7.1f} | Mean20 R: {mean_score:7.1f} | FPS: {fps:.1f}")

        f_indices_episode.append(f_idx)
        metrics['episode_rewards'].append(ep_r)
        metrics['mean_rewards'].append(mean_score)
        metrics['episode_steps'].append(steps)
        metrics['epsilons'].append(agent.get_epsilon(f_idx))

        if ep > 0 and ep % save_interval == 0:
             save_path = os.path.join(log_dir, f"mario_munchausen_ep{ep}.pth")
             print(f"\nSaving checkpoint to {save_path}")
             torch.save(agent.q.state_dict(), save_path)

        if ep > 0 and ep % plot_interval == 0:
            print(f"\nGenerating plots at episode {ep}...")
            save_plots(log_dir, metrics, f_indices_episode, f_indices_loss)
            print("Plots saved.")


    final_save_path = os.path.join(log_dir, f"mario_munchausen_final.pth")
    print(f"Training finished. Saving final model to {final_save_path}")
    torch.save(agent.q.state_dict(), final_save_path)

    print("Generating final plots...")
    save_plots(log_dir, metrics, f_indices_episode, f_indices_loss)
    print("Final plots saved.")

    env.close()

if __name__=='__main__':
    train()