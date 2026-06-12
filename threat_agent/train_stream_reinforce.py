#!/usr/bin/env python3
"""Train REINFORCE on StreamThreatEnv."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from threat_agent.stream_env import StreamEnvConfig, StreamThreatEnv
from threat_agent.stream_eval import StreamEval


class PolicyNet(nn.Module):
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


def eval_policy(policy: PolicyNet, env: StreamThreatEnv, episodes: int, device: torch.device):
    policy.eval()
    ev = StreamEval(labels=env.labels)
    with torch.no_grad():
        for _ in range(episodes):
            s, info = env.reset()
            done = False
            ep_return = 0.0
            steps = 0
            pred, true = None, None
            declared_step = None
            detection_delay = None
            while not done:
                s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                dist = torch.distributions.Categorical(logits=policy(s_t))
                a = int(torch.argmax(dist.logits, dim=1).item())
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
                pred, true = "benign", "benign"
            ev.add(true, pred, steps, declared_step, ep_return, info.get("first_attack_pos"), detection_delay)
    return ev.summary()


def discounted_returns(rewards: list[float], gamma: float):
    out = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return list(reversed(out))


def parse_args():
    p = argparse.ArgumentParser(description="Train stream REINFORCE.")
    p.add_argument("--stream-data", type=Path, default=Path("results/stream_events.ndjson"))
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--window-size", type=int, default=25)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--eval-episodes", type=int, default=80)
    p.add_argument("--save-model", type=Path, default=Path("checkpoints/stream_reinforce.pt"))
    p.add_argument("--metrics-output", type=Path, default=Path("results/stream_reinforce_metrics.json"))
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

    policy = PolicyNet(train_env.state_size, train_env.action_size).to(device)
    opt = optim.Adam(policy.parameters(), lr=args.lr)

    for ep in range(1, args.episodes + 1):
        s, _ = train_env.reset()
        done = False
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        entropies: list[torch.Tensor] = []
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            dist = torch.distributions.Categorical(logits=policy(s_t))
            a = int(dist.sample().item())
            ns, r, terminated, truncated, _ = train_env.step(a)
            log_probs.append(dist.log_prob(torch.tensor(a, device=device)))
            entropies.append(dist.entropy().squeeze(0))
            rewards.append(float(r))
            s = ns
            done = terminated or truncated

        rets = discounted_returns(rewards, args.gamma)
        ret_t = torch.tensor(rets, dtype=torch.float32, device=device)
        if ret_t.numel() > 1:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std(unbiased=False) + 1e-8)
        loss = torch.stack([-(lp * rt) for lp, rt in zip(log_probs, ret_t)]).sum()
        if entropies:
            loss += torch.stack([-(args.entropy_coef * e) for e in entropies]).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % args.eval_every == 0:
            val = eval_policy(policy, val_env, args.eval_episodes, device)
            print(f"Episode {ep} val={val}")

    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), args.save_model)
    print(f"Saved model: {args.save_model.resolve()}")
    val = eval_policy(policy, val_env, args.eval_episodes, device)
    test = eval_policy(policy, test_env, args.eval_episodes, device)
    print(f"val:  {val}")
    print(f"test: {test}")
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(
            {"algorithm": "stream_reinforce", "val": val, "test": test, "episodes": args.episodes, "seed": args.seed},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics: {args.metrics_output.resolve()}")


if __name__ == "__main__":
    main()

