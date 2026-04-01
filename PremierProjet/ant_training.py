"""
Ant-v5 — PPO training with real-time 4×2 mosaic visualization.
Training runs on GPU (CUDA), 8 visual envs rendered live via pygame.
"""

from __future__ import annotations

import time
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pygame
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# ── Config ─────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 5_000_000
N_TRAIN_ENVS    = 16          # parallel envs for training (no render)
N_VIS_ENVS      = 8           # envs rendered in the mosaic
VIS_COLS        = 4
VIS_ROWS        = 2
FRAME_W         = 480         # width of each cell in px
FRAME_H         = 360         # height of each cell in px
STATS_BAR_H     = 64
VIS_STEPS       = 4           # steps run on visual envs each callback
VIS_FREQ        = 512         # training steps between display refreshes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Palette
C = {
    "bg":    (12,  12,  18),
    "panel": (22,  22,  34),
    "sep":   (40,  40,  60),
    "white": (230, 230, 230),
    "gray":  (110, 110, 135),
    "blue":  (80,  170, 240),
    "green": (75,  220, 130),
    "red":   (220, 80,  80),
    "gold":  (255, 200, 50),
}
# ───────────────────────────────────────────────────────────────────────────────


class MosaicCallback(BaseCallback):
    """
    Custom SB3 callback that renders 8 Ant-v5 environments in a pygame mosaic
    while training progresses in the background.
    """

    def __init__(self, n_vis: int = N_VIS_ENVS, freq: int = VIS_FREQ, verbose: int = 0):
        super().__init__(verbose)
        self.n_vis       = n_vis
        self.freq        = freq
        self.vis_envs:   List[gym.Env]      = []
        self.obs_list:   List[np.ndarray]   = []
        self.ep_rewards: List[float]        = [0.0] * n_vis
        self.ep_counts:  List[int]          = [0]   * n_vis
        self.screen: Optional[pygame.Surface] = None
        self.font_md: Optional[pygame.font.Font] = None
        self.font_sm: Optional[pygame.font.Font] = None
        self.clock  = pygame.time.Clock()
        self.t_start = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _init_callback(self) -> None:
        self.t_start = time.time()

        # Spawn visual environments
        for _ in range(self.n_vis):
            env = gym.make("Ant-v5", render_mode="rgb_array")
            obs, _ = env.reset()
            self.vis_envs.append(env)
            self.obs_list.append(obs)

        # Init pygame window
        pygame.init()
        win_w = VIS_COLS * FRAME_W
        win_h = VIS_ROWS * FRAME_H + STATS_BAR_H
        self.screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Ant-v5  —  Live Training Mosaic")
        self.font_md = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_sm = pygame.font.SysFont("monospace", 12)

    def _on_step(self) -> bool:
        # Forward pygame close event even between renders
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        if self.n_calls % self.freq == 0:
            self._refresh()
        return True

    def _on_training_end(self) -> None:
        for env in self.vis_envs:
            env.close()
        pygame.quit()

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Step visual envs, capture frames, build and blit mosaic."""
        frames: List[np.ndarray] = []

        for i, env in enumerate(self.vis_envs):
            for _ in range(VIS_STEPS):
                action, _ = self.model.predict(self.obs_list[i], deterministic=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                self.ep_rewards[i] += float(reward)

                if terminated or truncated:
                    self.ep_counts[i] += 1
                    self.ep_rewards[i]  = 0.0
                    obs, _ = env.reset()

                self.obs_list[i] = obs

            raw   = env.render()                                 # H×W×3 uint8
            frame = _resize(raw, FRAME_W, FRAME_H)
            frame = self._burn_overlay(frame, i)
            frames.append(frame)

        self._blit_mosaic(frames)
        self._blit_stats_bar()
        pygame.display.flip()
        self.clock.tick(60)

    def _burn_overlay(self, frame: np.ndarray, idx: int) -> np.ndarray:
        """Add a semi-transparent top strip with agent index and episode reward."""
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        strip = pygame.Surface((FRAME_W, 26), pygame.SRCALPHA)
        strip.fill((0, 0, 0, 170))
        surf.blit(strip, (0, 0))

        episodes = self.ep_counts[idx]
        reward   = self.ep_rewards[idx]
        text     = f" Agent {idx + 1:02d}   ep {episodes:>4d}   reward {reward:+8.1f}"
        label    = self.font_md.render(text, True, C["gold"])
        surf.blit(label, (6, 5))

        return pygame.surfarray.array3d(surf).swapaxes(0, 1)

    def _blit_mosaic(self, frames: List[np.ndarray]) -> None:
        self.screen.fill(C["bg"])

        for idx, frame in enumerate(frames):
            col, row = idx % VIS_COLS, idx // VIS_COLS
            x,   y   = col * FRAME_W,  row * FRAME_H

            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            self.screen.blit(surf, (x, y))

            # Cell border
            pygame.draw.rect(self.screen, C["sep"], (x, y, FRAME_W, FRAME_H), 2)

    def _blit_stats_bar(self) -> None:
        """Bottom stats bar: progress, speed, ETA, device."""
        win_w   = VIS_COLS * FRAME_W
        bar_y   = VIS_ROWS * FRAME_H
        elapsed = time.time() - self.t_start
        sps     = self.num_timesteps / max(elapsed, 1)
        eta_s   = (TOTAL_TIMESTEPS - self.num_timesteps) / max(sps, 1)
        pct     = self.num_timesteps / TOTAL_TIMESTEPS * 100

        pygame.draw.rect(self.screen, C["panel"], (0, bar_y, win_w, STATS_BAR_H))

        line1 = (
            f"  Steps {self.num_timesteps:>9,} / {TOTAL_TIMESTEPS:,}"
            f"   {pct:.1f}%"
            f"   {sps:,.0f} sps"
        )
        line2 = (
            f"  Elapsed {elapsed / 60:.1f} min"
            f"   ETA {eta_s / 60:.1f} min"
            f"   Device: {DEVICE.upper()}"
            + (f"  [{torch.cuda.get_device_name(0)}]" if DEVICE == "cuda" else "")
        )

        self.screen.blit(self.font_md.render(line1, True, C["blue"]), (0, bar_y + 6))
        self.screen.blit(self.font_sm.render(line2, True, C["gray"]), (0, bar_y + 28))

        # Progress bar
        filled = int(win_w * pct / 100)
        pygame.draw.rect(self.screen, C["green"], (0, bar_y + STATS_BAR_H - 8, filled, 8))
        pygame.draw.rect(self.screen, C["sep"],   (filled, bar_y + STATS_BAR_H - 8, win_w - filled, 8))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fast nearest-neighbour resize without cv2."""
    src_h, src_w = frame.shape[:2]
    x_idx = (np.arange(w) * src_w / w).astype(np.int32)
    y_idx = (np.arange(h) * src_h / h).astype(np.int32)
    return frame[np.ix_(y_idx, x_idx)]


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device : {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    train_env = make_vec_env("Ant-v5", n_envs=N_TRAIN_ENVS)

    model = PPO(
        "MlpPolicy",
        train_env,
        device=DEVICE,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=MosaicCallback())

    model.save("ant_v5_model")
    print("Model saved  →  ant_v5_model.zip")

    train_env.close()


if __name__ == "__main__":
    main()
