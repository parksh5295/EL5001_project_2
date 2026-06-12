#!/usr/bin/env python3
"""Train a lightweight DQN agent for Threat Investigation Agent."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from threat_agent.env import EnvConfig, ThreatInvestigationEnv
from threat_agent.metrics import EvalAccumulator


class QNet(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    done: float
    mask: np.ndarray
    nmask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, tr: Transition):
        self.buf.append(tr)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


def masked_argmax(q: np.ndarray, mask: np.ndarray) -> int:
    q2 = q.copy()
    q2[mask <= 0.0] = -1e9
    return int(np.argmax(q2))


def evaluate(policy_net: QNet, env: ThreatInvestigationEnv, episodes: int, device: torch.device):
    policy_net.eval()
    acc = EvalAccumulator(labels=env.tactics)
    with torch.no_grad():
        for _ in range(episodes):
            s, info = env.reset()
            done = False
            ep_return = 0.0
            ep_steps = 0
            declared_step = None
            pred = None
            true = info["tactic"]
            while not done:
                mask = np.array(info["action_mask"], dtype=np.float32)
                q = policy_net(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))[0].cpu().numpy()
                a = masked_argmax(q, mask)
                s, r, terminated, truncated, info = env.step(a)
                ep_return += r
                ep_steps += 1
                if info.get("declared_tactic") is not None and declared_step is None:
                    pred = info.get("declared_tactic")
                    declared_step = ep_steps
                done = terminated or truncated
            acc.add(true_label=true, pred_label=pred, steps=ep_steps, declared_step=declared_step, episode_return=ep_return)
    return acc.summary()


def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on Threat Investigation Agent environment.")
    p.add_argument("--dataset", type=Path, default=Path("results/threat_agent_data.json"))
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--target-update", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=80)
    p.add_argument("--save-model", type=Path, default=Path("checkpoints/threat_agent_dqn.pt"))
    p.add_argument("--metrics-output", type=Path, default=Path("results/dqn_metrics.json"))
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EnvConfig(
        max_steps=args.max_steps,
        investigate_cost=0.5,
        invalid_action_penalty=0.25,
        correct_declare_reward=10.0,
        wrong_declare_penalty=10.0,
        reveal_success_prob=0.9,  # stochastic reveal to satisfy project condition
        seed=args.seed,
    )
    train_env = ThreatInvestigationEnv(args.dataset, split="train", config=cfg)
    val_env = ThreatInvestigationEnv(args.dataset, split="val", config=cfg)
    test_env = ThreatInvestigationEnv(args.dataset, split="test", config=cfg)

    policy_net = QNet(train_env.state_size, train_env.action_size).to(device)
    target_net = QNet(train_env.state_size, train_env.action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    opt = optim.Adam(policy_net.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer_size)
    epsilon = args.epsilon_start
    global_step = 0

    for ep in range(1, args.episodes + 1):
        s, info = train_env.reset()
        done = False
        while not done:
            mask = np.array(info["action_mask"], dtype=np.float32)
            if np.random.rand() < epsilon:
                valid = np.where(mask > 0.0)[0]
                a = int(np.random.choice(valid))
            else:
                with torch.no_grad():
                    q = policy_net(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))[0].cpu().numpy()
                a = masked_argmax(q, mask)

            ns, r, terminated, truncated, ninfo = train_env.step(a)
            nmask = np.array(ninfo["action_mask"], dtype=np.float32)
            done = terminated or truncated
            buf.add(Transition(s, a, r, ns, 1.0 if done else 0.0, mask, nmask))
            s, info = ns, ninfo
            global_step += 1

            if len(buf) >= args.batch_size:
                batch = buf.sample(args.batch_size)
                s_b = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32, device=device)
                a_b = torch.tensor([b.a for b in batch], dtype=torch.long, device=device).unsqueeze(1)
                r_b = torch.tensor([b.r for b in batch], dtype=torch.float32, device=device)
                ns_b = torch.tensor(np.stack([b.ns for b in batch]), dtype=torch.float32, device=device)
                d_b = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device)
                nmask_b = torch.tensor(np.stack([b.nmask for b in batch]), dtype=torch.float32, device=device)

                q_pred = policy_net(s_b).gather(1, a_b).squeeze(1)

                with torch.no_grad():
                    q_next = target_net(ns_b)
                    q_next = q_next.masked_fill(nmask_b <= 0, -1e9)
                    q_target = r_b + (1.0 - d_b) * args.gamma * torch.max(q_next, dim=1).values

                loss = nn.functional.mse_loss(q_pred, q_target)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if global_step % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if ep % args.eval_every == 0:
            val_result = evaluate(policy_net, val_env, args.eval_episodes, device)
            print(f"Episode {ep} epsilon={epsilon:.3f} val={val_result}")

    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_net.state_dict(), args.save_model)
    print(f"Saved model: {args.save_model.resolve()}")

    val_final = evaluate(policy_net, val_env, args.eval_episodes, device)
    test_final = evaluate(policy_net, test_env, args.eval_episodes, device)
    print("Final evaluation")
    print(f"val:  {val_final}")
    print(f"test: {test_final}")

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(
            {
                "algorithm": "dqn",
                "val": val_final,
                "test": test_final,
                "episodes": args.episodes,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics: {args.metrics_output.resolve()}")


if __name__ == "__main__":
    main()
