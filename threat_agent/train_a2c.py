#!/usr/bin/env python3
"""Train A2C baseline for Threat Investigation Agent."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from threat_agent.env import EnvConfig, ThreatInvestigationEnv


class ActorCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_size)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor):
    return torch.distributions.Categorical(logits=logits.masked_fill(mask <= 0, -1e9))


def evaluate(model: ActorCritic, env: ThreatInvestigationEnv, episodes: int, device: torch.device):
    model.eval()
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
                logits, _ = model(s_t)
                dist = masked_categorical(logits, mask)
                action = int(torch.argmax(dist.logits, dim=1).item())
                s, r, terminated, truncated, info = env.step(action)
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
    p = argparse.ArgumentParser(description="Train A2C on Threat Investigation Agent.")
    p.add_argument("--dataset", type=Path, default=Path("results/threat_agent_data.json"))
    p.add_argument("--episodes", type=int, default=2500)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=80)
    p.add_argument("--save-model", type=Path, default=Path("checkpoints/threat_agent_a2c.pt"))
    p.add_argument("--metrics-output", type=Path, default=Path("results/a2c_metrics.json"))
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

    model = ActorCritic(train_env.state_size, train_env.action_size).to(device)
    optim_ac = optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.episodes + 1):
        s, info = train_env.reset()
        done = False
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        entropies: list[torch.Tensor] = []
        dones: list[float] = []

        while not done:
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(s_t)
            dist = masked_categorical(logits, mask)
            action = int(dist.sample().item())

            ns, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated

            log_probs.append(dist.log_prob(torch.tensor(action, device=device)))
            values.append(value.squeeze(0))
            rewards.append(float(reward))
            entropies.append(dist.entropy().squeeze(0))
            dones.append(1.0 if done else 0.0)
            s = ns

        # bootstrap value at final state (0 for terminal)
        returns = []
        g = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + args.gamma * g * (1.0 - d)
            returns.append(g)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.stack(values)
        logp_t = torch.stack(log_probs)
        entropy_t = torch.stack(entropies)

        advantages = returns_t - values_t
        actor_loss = -(logp_t * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy_t.mean()

        optim_ac.zero_grad()
        loss.backward()
        optim_ac.step()

        if ep % args.eval_every == 0:
            val_result = evaluate(model, val_env, args.eval_episodes, device)
            print(f"Episode {ep} val={val_result}")

    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_model)
    print(f"Saved model: {args.save_model.resolve()}")
    val_final = evaluate(model, val_env, args.eval_episodes, device)
    test_final = evaluate(model, test_env, args.eval_episodes, device)
    print("Final evaluation")
    print(f"val:  {val_final}")
    print(f"test: {test_final}")

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(
            {
                "algorithm": "a2c",
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

