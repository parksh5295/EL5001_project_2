#!/usr/bin/env python3
"""Train DQN on StreamThreatEnv (window observation + declare actions)."""

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

from threat_agent.stream_env import StreamEnvConfig, StreamThreatEnv
from threat_agent.stream_eval import StreamEval


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
class Tr:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    done: float


class Buffer:
    def __init__(self, cap: int):
        self.buf = deque(maxlen=cap)

    def add(self, tr: Tr):
        self.buf.append(tr)

    def sample(self, bs: int):
        idx = np.random.choice(len(self.buf), size=bs, replace=False)
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


def eval_policy(net: QNet, env: StreamThreatEnv, episodes: int, device: torch.device):
    net.eval()
    ev = StreamEval(labels=env.labels)
    with torch.no_grad():
        for _ in range(episodes):
            s, info = env.reset()
            done = False
            ep_return = 0.0
            steps = 0
            pred = None
            true = None
            declared_step = None
            detection_delay = None
            while not done:
                q = net(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))[0].cpu().numpy()
                a = int(np.argmax(q))
                s, r, terminated, truncated, info = env.step(a)
                ep_return += r
                steps += 1
                if info.get("declared_label") is not None:
                    pred = info["declared_label"]
                    true = info["true_label_at_declare"]
                    declared_step = info["declared_step"]
                    detection_delay = info.get("detection_delay")
                done = terminated or truncated
            if pred is None:
                pred = "benign"
                true = "benign"
            ev.add(
                true_label=true,
                pred_label=pred,
                steps=steps,
                declared_step=declared_step,
                episode_return=ep_return,
                first_attack_pos=info.get("first_attack_pos"),
                detection_delay=detection_delay,
            )
    return ev.summary()


def parse_args():
    p = argparse.ArgumentParser(description="Train stream DQN.")
    p.add_argument("--stream-data", type=Path, default=Path("results/stream_events.ndjson"))
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--window-size", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--target-update", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--eval-episodes", type=int, default=80)
    p.add_argument("--save-model", type=Path, default=Path("checkpoints/stream_dqn.pt"))
    p.add_argument("--metrics-output", type=Path, default=Path("results/stream_dqn_metrics.json"))
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = StreamEnvConfig(window_size=args.window_size, max_steps=args.max_steps, seed=args.seed)
    train_env = StreamThreatEnv(args.stream_data, split="train", config=cfg)
    val_env = StreamThreatEnv(args.stream_data, split="val", config=cfg)
    test_env = StreamThreatEnv(args.stream_data, split="test", config=cfg)

    net = QNet(train_env.state_size, train_env.action_size).to(device)
    target = QNet(train_env.state_size, train_env.action_size).to(device)
    target.load_state_dict(net.state_dict())
    target.eval()
    opt = optim.Adam(net.parameters(), lr=args.lr)
    buf = Buffer(args.buffer_size)
    eps = args.epsilon_start
    gstep = 0

    for ep in range(1, args.episodes + 1):
        s, _ = train_env.reset()
        done = False
        while not done:
            if np.random.rand() < eps:
                a = np.random.randint(train_env.action_size)
            else:
                with torch.no_grad():
                    q = net(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))[0].cpu().numpy()
                a = int(np.argmax(q))

            ns, r, terminated, truncated, _ = train_env.step(a)
            done = terminated or truncated
            buf.add(Tr(s, a, r, ns, 1.0 if done else 0.0))
            s = ns
            gstep += 1

            if len(buf) >= args.batch_size:
                batch = buf.sample(args.batch_size)
                s_b = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32, device=device)
                a_b = torch.tensor([b.a for b in batch], dtype=torch.long, device=device).unsqueeze(1)
                r_b = torch.tensor([b.r for b in batch], dtype=torch.float32, device=device)
                ns_b = torch.tensor(np.stack([b.ns for b in batch]), dtype=torch.float32, device=device)
                d_b = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device)

                q_pred = net(s_b).gather(1, a_b).squeeze(1)
                with torch.no_grad():
                    q_next = target(ns_b).max(dim=1).values
                    q_tar = r_b + (1.0 - d_b) * args.gamma * q_next
                loss = nn.functional.mse_loss(q_pred, q_tar)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if gstep % args.target_update == 0:
                target.load_state_dict(net.state_dict())

        eps = max(args.epsilon_end, eps * args.epsilon_decay)
        if ep % args.eval_every == 0:
            val = eval_policy(net, val_env, args.eval_episodes, device)
            print(f"Episode {ep} eps={eps:.3f} val={val}")

    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.save_model)
    print(f"Saved model: {args.save_model.resolve()}")

    val = eval_policy(net, val_env, args.eval_episodes, device)
    test = eval_policy(net, test_env, args.eval_episodes, device)
    print(f"val:  {val}")
    print(f"test: {test}")

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(
            {
                "algorithm": "stream_dqn",
                "val": val,
                "test": test,
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

