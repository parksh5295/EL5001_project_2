#!/usr/bin/env python3
"""Train A2C on StreamThreatEnv."""

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


def masked_logits(logits: torch.Tensor, action_mask: np.ndarray, device: torch.device):
    mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    return logits.masked_fill(~mask_t, -1e9)


def eval_policy(model: ActorCritic, env: StreamThreatEnv, episodes: int, device: torch.device):
    model.eval()
    ev = StreamEval(labels=env.labels)
    with torch.no_grad():
        for _ in range(episodes):
            s, _ = env.reset()
            done = False
            ep_return = 0.0
            steps = 0
            final_info = {}
            while not done:
                s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(s_t)
                logits = masked_logits(logits, env.get_action_mask(), device)
                dist = torch.distributions.Categorical(logits=logits)
                a = int(torch.argmax(dist.logits, dim=1).item())
                s, r, terminated, truncated, info = env.step(a)
                ep_return += r
                steps += 1
                final_info = info
                done = terminated or truncated
            ev.add_segment_episode(
                pred_trace=final_info.get("episode_pred_trace", []),
                gt_trace=final_info.get("episode_gt_trace", []),
                steps=steps,
                episode_return=ep_return,
                boundary_tp=final_info.get("boundary_tp", 0),
                boundary_fp=final_info.get("boundary_fp", 0),
                boundary_fn=final_info.get("boundary_fn", 0),
                action_counts=final_info.get("episode_action_counts", {}),
                reward_terms=final_info.get("episode_reward_terms", {}),
                attack_step_stats=final_info.get("episode_attack_step_stats", {}),
            )
    return ev.summary()


def print_seglog(tag: str, metric: dict):
    if not metric:
        return
    print(
        "[SEGLOG] {} "
        "micro_f1={:.4f} boundary_f1={:.4f} "
        "wait={:.3f}(u={:.3f},h={:.3f}) start={:.3f} end={:.3f} invalid={:.3f} "
        "atk_cov={:.3f} atk_hit={:.3f}".format(
            tag,
            float(metric.get("event_micro_f1", 0.0)),
            float(metric.get("segment_boundary_f1", 0.0)),
            float(metric.get("action_wait_ratio", 0.0)),
            float(metric.get("action_wait_unsure_ratio", 0.0)),
            float(metric.get("action_hold_active_ratio", 0.0)),
            float(metric.get("action_start_ratio", 0.0)),
            float(metric.get("action_end_ratio", 0.0)),
            float(metric.get("invalid_action_ratio", 0.0)),
            float(metric.get("attack_step_pred_coverage", 0.0)),
            float(metric.get("attack_step_overlap_hit", 0.0)),
        )
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train stream A2C.")
    p.add_argument("--stream-data", type=Path, default=Path("results/stream_events.ndjson"))
    p.add_argument("--train-stream-data", type=Path, default=None)
    p.add_argument("--val-stream-data", type=Path, default=None)
    p.add_argument("--test-stream-data", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--window-size", type=int, default=25)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--eval-episodes", type=int, default=80)
    p.add_argument("--save-model", type=Path, default=Path("checkpoints/stream_a2c.pt"))
    p.add_argument("--metrics-output", type=Path, default=Path("results/stream_a2c_metrics.json"))
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = StreamEnvConfig(window_size=args.window_size, max_steps=args.max_steps, seed=args.seed)
    train_path = args.train_stream_data or args.stream_data
    val_path = args.val_stream_data or args.stream_data
    test_path = args.test_stream_data or args.stream_data
    train_env = StreamThreatEnv(train_path, split="train", config=cfg)
    val_env = StreamThreatEnv(val_path, split="val", config=cfg, tactics=train_env.tactics)
    test_env = StreamThreatEnv(test_path, split="test", config=cfg, tactics=train_env.tactics)

    model = ActorCritic(train_env.state_size, train_env.action_size).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.episodes + 1):
        s, _ = train_env.reset()
        done = False
        logps = []
        vals = []
        rewards = []
        ents = []
        dones = []
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(s_t)
            logits = masked_logits(logits, train_env.get_action_mask(), device)
            dist = torch.distributions.Categorical(logits=logits)
            a = int(dist.sample().item())
            ns, r, terminated, truncated, _ = train_env.step(a)
            done = terminated or truncated
            logps.append(dist.log_prob(torch.tensor(a, device=device)))
            vals.append(value.squeeze(0))
            rewards.append(float(r))
            ents.append(dist.entropy().squeeze(0))
            dones.append(1.0 if done else 0.0)
            s = ns

        returns = []
        g = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + args.gamma * g * (1.0 - d)
            returns.append(g)
        returns_t = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=device)
        vals_t = torch.stack(vals)
        adv = returns_t - vals_t
        actor_loss = -(torch.stack(logps) * adv.detach()).mean()
        critic_loss = adv.pow(2).mean()
        loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * torch.stack(ents).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % args.eval_every == 0:
            val = eval_policy(model, val_env, args.eval_episodes, device)
            print(f"Episode {ep} val={val}")
            print_seglog(f"a2c/val/ep{ep}", val)

    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_model)
    print(f"Saved model: {args.save_model.resolve()}")
    val = eval_policy(model, val_env, args.eval_episodes, device)
    test = eval_policy(model, test_env, args.eval_episodes, device)
    print(f"val:  {val}")
    print(f"test: {test}")
    print_seglog("a2c/val/final", val)
    print_seglog("a2c/test/final", test)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(
            {"algorithm": "stream_a2c", "val": val, "test": test, "episodes": args.episodes, "seed": args.seed},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics: {args.metrics_output.resolve()}")


if __name__ == "__main__":
    main()

