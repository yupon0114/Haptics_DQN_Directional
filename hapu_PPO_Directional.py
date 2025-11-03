# -*- coding: utf-8 -*-
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import serial
import time
import re
import subprocess
import csv
import hapu_phase_module

# -----------------------------
# Training Configuration
# -----------------------------
verbose = 0
episodes = 100
timesteps = 200
total = episodes * timesteps
learning_rate = 0.001  #ここを変える①
gamma = 0.3            #ここを変える②
clip_range = 0.5       #ここを変える③
threshold_deg = 20.0
patience = 10

port = '/dev/ttyACM0'
ser = serial.Serial(port, 115200)

# ==== 初期状態 ====
phase = np.array([0.0]*8)
target_intensity = 0
best_phases = 0
best_intensities = 0


def kakikae(phase):
    hapu_phase_module.phase_set(phase)
    
def speakerON(A):
    hapu_phase_module.start_output(A)
    ser.write("READ")

# -----------------------------
# max_phase determination
# -----------------------------
def determine_max_phase(source_num):
    aa = [90, 180, 270, 360]
    pressures = []
    for deg in aa:
        rad = np.deg2rad(deg)
        phase[source_num] = rad
        kakikae(phase)
        hapu_phase_module.start_output(source_num) # ON
        readings = [float(ser.readline().decode('utf-8').strip()) for _ in range(5)]
        intensity = max(readings)
        pressures.append(intensity)
        
    plus_pressures = [
        pressures[0] + pressures[1],
        pressures[1] + pressures[2],
        pressures[2] + pressures[3],
        pressures[3] + pressures[0]
    ]
    max_pressures = max(plus_pressures)
    if max_pressures == plus_pressures[0]:
        max_phase = 135 if pressures[0] > pressures[1] else 180
    elif max_pressures == plus_pressures[1]:
        max_phase = 225 if pressures[1] > pressures[2] else 270
    elif max_pressures == plus_pressures[2]:
        max_phase = 315 if pressures[2] > pressures[3] else 360
    else:
        max_phase = 45 if pressures[3] > pressures[0] else 90
    print(f"speaker{source_num},max:{max_phase}°")
    return np.deg2rad(max_phase)

max_phase_list = [determine_max_phase(i) for i in range(1, 8)]

# -----------------------------
# Environment Definition
# -----------------------------
class MultiSourcePhaseEnv(gym.Env):
    def __init__(self, speaker_intensity):
        super(MultiSourcePhaseEnv, self).__init__()
        self.max_phase_list = max_phase_list
        self.action_space = spaces.Box(low = -np.pi/8, high = np.pi/8, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low = 0, high = np.inf, shape=(1,), dtype=np.float32)
        self.phases = phase
        self.source_indices = list(range(1, 8))
#         self.focus_point = focus_point
#         self.sound_sources = sound_sources
#         self.wavelength = wavelength
#         self.initial_pressure = initial_pressure
        self.current_step = 0
        self.best_intensity = -np.inf
        self.best_phases = np.zeros(7)
        self.np_random = np.random.default_rng()
        self.episode_rewards = []
        self.reward_sum = 0
        self.episode_cnt = 0
        self.speaker_intensity = speaker_intensity
        self.phase_changes = []
        self.action_deg = None
        
    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
        self.reward_sum = 0
        self.episode_cnt += 1
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        self.action_deg = np.rad2deg(action)
        self.phase_changes.append(self.action_deg)
        for i, idx in enumerate(self.source_indices):
            self.phases[idx] = self.max_phase_list[i] + action[i]
        
        intensity = self._get_observation()
        self.reward_sum += intensity[0]
        
        if intensity[0] > self.best_intensity:
            self.best_intensity = intensity[0]
            self.best_phases = self.phases[1:8].copy()

        done = self.current_step == timesteps
        if done:
            self.episode_rewards.append(self.reward_sum)

        info = {
            "phases": self.phases[1:8],
            "intensity": intensity[0],
            "best_intensity": self.best_intensity,
            "best_phases": self.best_phases,
            "speaker_rewards": self.episode_rewards
        }
        return intensity, intensity[0], done, info

    def _get_observation(self):
        total_intensity = 0.0
        kakikae(self.phases)
        speakerON(9)
        readings = [float(ser.readline().decode('utf-8').strip()) for _ in range(5)]
        print(f"Step{self.current_step}: phase = {self.phases}°")
        print(f"readings:{max(readings):.2f}")
        total_intensity = max(readings)
#         for i in range(8):
#             total_intensity += propagation_model(
# #                 self.sound_sources[i],
#                 self.focus_point,
#                 self.wavelength,
#                 self.phases[i]
#             )
        return np.array([total_intensity], dtype=np.float32)

# -----------------------------
# Phase Stagnation Callback
# -----------------------------
class PhaseStagnationCallback(BaseCallback):
    def __init__(self, threshold_deg=threshold_deg, patience=patience, verbose=1):
        super().__init__(verbose)
        self.threshold_rad = np.deg2rad(threshold_deg)
        self.patience = patience
        self.stagnation_count = 0
        self.prev_phases = None

    def _on_step(self) -> bool:
        if "infos" in self.locals and len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            current_phases = info.get("phases", None)
            if current_phases is not None:
                if self.prev_phases is not None:
                    phase_diff = np.abs(current_phases - self.prev_phases)
                    if np.all(phase_diff < self.threshold_rad):
                        self.stagnation_count += 1
                    else:
                        self.stagnation_count = 0
                self.prev_phases = current_phases.copy()

                if self.stagnation_count >= self.patience:
                    if self.verbose:
                        print(f"Early stopping triggered due to phase stagnation at step {self.num_timesteps}")
                    return False
        return True

# -----------------------------
# Training Execution
# -----------------------------

# 目標強度を測定
speakerON(1)
speaker_intensity = max(float(ser.readline().decode('utf-8').strip()) for _ in range(5))
print(f"Speaker intensity: {speaker_intensity:.2f}")
hapu_phase_module.stop_output()

env = make_vec_env(lambda: MultiSourcePhaseEnv(speaker_intensity), n_envs=1)
model = PPO('MlpPolicy', env, learning_rate=learning_rate, verbose=verbose,
            clip_range=clip_range, gamma=gamma, n_steps=timesteps)
# env = make_vec_env(lambda: MultiSourcePhaseEnv(), n_envs=1)
callback = PhaseStagnationCallback(threshold_deg=threshold_deg, patience=patience, verbose=1)
model.learn(total_timesteps=total, callback=callback)

# -----------------------------
# Results and Visualization
# -----------------------------
final_env = env.envs[0]
raw_env = final_env.unwrapped

# plt.figure(figsize=(12, 6))
# plt.plot(raw_env.episode_rewards, label='Total Reward')
# plt.xlabel('Episodes')
# plt.ylabel('Rewards')
# plt.title('Learning Progress')
# plt.legend()
# plt.grid(True)
# plt.show()
# 
# plt.figure(figsize=(12, 6))
# for i in range(7):
#     phase_deg = [step[i] for step in raw_env.phase_changes]
#     plt.plot(phase_deg, label=f'Source {i+1}')
# plt.xlabel('Steps')
# plt.ylabel('Phase (degrees)')
# plt.title('Phase Changes per Source')
# plt.legend()
# plt.grid(True)
# plt.show()

final_phases_deg = np.rad2deg(raw_env.phases)
final_intensity = raw_env._get_observation()[0]
print("Final Phases (degrees):", np.round(final_phases_deg, 2))
print("Final Intensity (at final phases):", final_intensity)
print("Best Intensity:", raw_env.best_intensity)
print("Best Phases (degrees):", np.round(np.rad2deg(raw_env.best_phases), 2))


# 最後のリセット
print(phase)
speakerON(9)