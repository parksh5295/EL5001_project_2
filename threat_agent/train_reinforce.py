#!/usr/bin/env python3
"""Train a policy-gradient agent (REINFORCE) for Threat Investigation Agent."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from threat_agent.env import EnvConfig, ThreatInvestigationEnv


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


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor):
    masked_logits = logits.masked_fill(mask <= 0, -1e9)
    return torch.distributions.Categorical(logits=masked_logits)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return list(reversed(out))


def evaluate(policy: PolicyNet, env: ThreatInvestigationEnv, episodes: int, device: torch.device):
    policy.eval()
    correct = 0
    total_reward = 0.0
    total_steps = 0
    with torch.no_grad():
        for _ in range(episodes):
            s, info = env.reset()
            done = False
            while not done:
                s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
                dist = masked_categorical(policy(s_t), mask)
                a = int(torch.argmax(dist.logits, dim=1).item())
                s, r, terminated, truncated, info = env.step(a)
                total_reward += r
                total_steps += 1
                done = terminated or truncated
            if info.get("correct"):
                correct += 1
    return {
        "accuracy": correct / episodes if episodes else 0.0,
        "avg_return": total_reward / episodes if episodes else 0.0,
        "avg_steps": total_steps / episodes if episodes else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train REINFORCE on Threat Investigation Agent.")
    p.add_argument("--dataset", type=Path, default=Path("results/threat_agent_data.json"))
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--eval-episodes", type=int, default=80)
    p.add_argument("--save-model", type=Path, default=Path("checkpoints/threat_agent_reinforce.pt"))
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
        reveal_success_prob=0.9,
        seed=args.seed,
    )
    train_env = ThreatInvestigationEnv(args.dataset, split="train", config=cfg)
    val_env = ThreatInvestigationEnv(args.dataset, split="val", config=cfg)
    test_env = ThreatInvestigationEnv(args.dataset, split="test", config=cfg)

    policy = PolicyNet(train_env.state_size, train_env.action_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    for ep in range(1, args.episodes + 1):
        s, info = train_env.reset()
        done = False
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        entropies: list[torch.Tensor] = []

        while not done:
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
            dist = masked_categorical(policy(s_t), mask)
            action = int(dist.sample().item())

            ns, r, terminated, truncated, info = train_env.step(action)
            log_probs.append(dist.log_prob(torch.tensor(action, device=device)))
            entropies.append(dist.entropy())
            rewards.append(float(r))
            s = ns
            done = terminated or truncated

        returns = discounted_returns(rewards, args.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        # Normalize returns for lower variance (safe for short trajectories)
        if returns_t.numel() > 1:
            std = returns_t.std(unbiased=False)
            returns_t = (returns_t - returns_t.mean()) / (std + 1e-8)

        policy_loss = []
        entropy_loss = []
        for lp, ret, ent in zip(log_probs, returns_t, entropies):
            policy_loss.append(-lp * ret)
            entropy_loss.append(-args.entropy_coef * ent)
        loss = torch.stack(policy_loss).sum() + torch.stack(entropy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % args.eval_every == 0:
            val_result = evaluate(policy, val_env, args.eval_episodes, device)
            print(f"Episode {ep} val={val_result}")

    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), args.save_model)
    print(f"Saved model: {args.save_model.resolve()}")

    print("Final evaluation")
    print(f"val:  {evaluate(policy, val_env, args.eval_episodes, device)}")
    print(f"test: {evaluate(policy, test_env, args.eval_episodes, device)}")


if __name__ == "__main__":
    main()
