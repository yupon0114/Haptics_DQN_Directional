# -*- coding: utf-8 -*-
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use('Agg')  # Pi の非GUI環境でも保存可
import matplotlib.pyplot as plt
import serial
import time
# import re
# import subprocess
import csv
import os
import hapu_phase_module
# import spidev
# import RPi.GPIO as GPIO


# ==== 設定 ====
exploration_initial_eps = 1.0
epsilon_decay_rate = 0.9
learning_rate = 0.001
gamma = 0.995
timesteps = 7
episodes = 20
learning_starts = 7
final_epsilon = 0.01
sep = 9
#ls /dev/ttyACM* konoko-dodeportkakuninnsite
port = '/dev/ttyACM1'
ser = serial.Serial(port, 115200)
# hapu_phase_module.gishiki()
# ino_sketch = "/home/pi/Desktop/sketchbook/libraries/copy1009/copy1009.ino"
# board_type = "arduino:avr:uno"

# ==== 初期状態 ====
phase = np.array([0] * 8)
point_str = 'point1'
target_intensity = 0
best_phases = 0
best_intensities = 0


def kakikae(phase):
    hapu_phase_module.phase_set(phase)
    
def speakerON(A):
    hapu_phase_module.start_output(A)

#     readings = [float(ser.readline().decode('utf-8').strip()) for _ in range(5)]
#     print(f"readings0: {max(readings):.2f}")
    time.sleep(0.5)
    
def get_voltage_from_pico(ser):
    print("sendREAD")
    ser.write(b'READ\n')  # Picoに測定要求を送信
    print("waitpico")
    time.sleep(0.5)
    line = ser.readline().decode('utf-8').strip()
    try:
        return float(line)
    except ValueError:
        print("ValueError")
        return 0.0
    
class PhaseOptimizationEnv(gym.Env):
    def __init__(self, source_num):
        super().__init__()
        self.action_space = spaces.Discrete(int(45 / sep))
        self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.float32)
        self.source_num = source_num
        self.reset()

    def reset(self):
        self.current_step = 0
        self.best_intensity = -np.inf
        self.reward_sum = 0
        self.episode_rewards = []
        self.phase_changes = []
        return 0

    def step(self, action):
        self.current_step += 1
        model._on_step()

        real_action = (max_phase - 45 + action * sep) % 360
        self.phase_changes.append(real_action)
        phase[self.source_num - 1] = real_action

        kakikae(phase)
        print(f"Step {self.current_step}: Source {self.source_num} = {real_action}°")
        
        self.sp = self._get_observation()
        #reward = self.sp / (target_intensity) # HOUSHU
        reward = self.sp
        self.reward_sum += reward
        self.episode_rewards.append(reward)

        if self.sp > self.best_intensity:
            self.best_intensity = self.sp
            self.best_phase = np.deg2rad(phase[self.source_num - 1])

        done = self.current_step >= timesteps
        info = {"phase": self.best_phase, "intensity": self.sp}
        return self.sp, reward, done, info

    def _get_observation(self):
        speakerON(self.source_num)
        ser.write(b"READ\n")
        readings = float(ser.readline().decode('utf-8').strip())
        print(f"readings: {readings:.2f}")
        #print("Readings:", readings, max(readings))
        hapu_phase_module.stop_output()
        return readings

    def seed(self, seed=None):
        import random, numpy as np
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

class CustomDQN(DQN):
    def __init__(self, *args, epsilon_decay_rate=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = exploration_initial_eps
        self.epsilon_decay_rate = epsilon_decay_rate

    def _on_step(self):
        self.epsilon = max(final_epsilon, self.epsilon * self.epsilon_decay_rate)
        return super()._on_step()


class PhaseStabilityCallback(BaseCallback):
    def __init__(self, env, stability_count=7, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.source_num = source_num
        self.stability_count = stability_count
        self.counter = 0
        self.last_phase = None

    def _on_step(self):
        current_phase = phase[self.source_num - 1]
        if self.last_phase != current_phase:
            self.counter = 1
            self.last_phase = current_phase
        else:
            self.counter += 1
        return self.counter < self.stability_count

# ==== メイン処理 ====
speakerON(1)

# 目標強度を測定
# print("A")
#ser.write(b"READ")
time.sleep(1)
# target_intensity = max(float(ser.readline().decode()) for _ in range(5))
# target_intensity = float(ser.readline().decode(utf-8))
target_intensity = get_voltage_from_pico(ser)
print(f"Target intensity: {target_intensity:.2f}")



hapu_phase_module.stop_output()
aa = [90, 180, 270, 360]

for source_num in range(2, 9):
    env = make_vec_env(lambda: PhaseOptimizationEnv(source_num), n_envs=1)

    speakerON(source_num)  # 'a'〜'g'

    pressures = []
    for angle in aa:
        phase[source_num - 1] = angle
        kakikae(phase)
        print(phase)
        speakerON(source_num)
        print(f"source{source_num}:{angle}",end = " ")
        speakerON(source_num)
        ser.write(b"READ\n")
#         readings = [float(ser.readline().decode('utf-8').strip()) for _ in range(5)]
        readings = float(ser.readline().decode('utf-8').strip())
        print(f"readings: {readings:.2f}")
        pressures.append(readings)
        
    # 位相推定
    plus_pressures = [
        pressures[0]+pressures[1],
        pressures[1]+pressures[2],
        pressures[2]+pressures[3],
        pressures[3]+pressures[0]
    ]
    if max(plus_pressures) == plus_pressures[0]:
        max_phase = 135 if pressures[0] > pressures[1] else 180
    elif max(plus_pressures) == plus_pressures[1]:
        max_phase = 225 if pressures[1] > pressures[2] else 270
    elif max(plus_pressures) == plus_pressures[2]:
        max_phase = 315 if pressures[2] > pressures[3] else 360
    else:
        max_phase = 45 if pressures[3] > pressures[0] else 90

    model = CustomDQN("MlpPolicy", env, train_freq=1, learning_starts=learning_starts,
                      learning_rate=learning_rate, gamma=gamma, verbose=0)
   
    callback = PhaseStabilityCallback(env.envs[0])
    model.learn(total_timesteps=timesteps * episodes, callback=callback)
    # 結果可視化
    final_env = env.envs[0].env
    rewards = final_env.episode_rewards
#     plt.figure()
#     plt.plot(rewards)
#     plt.title(f'Speaker {source_num} Rewards')
#     plt.savefig(f'/home/pi/Desktop/rewards/s{source_num}_reward.png')
# 
#     with open(f'/home/pi/Desktop/rewards/s{source_num}_reward.csv', 'w') as f:
#         csv.writer(f).writerow(rewards)
# 
#     print(f"Source {source_num}: Best Phase = {np.rad2deg(final_env.best_phase)%360:.2f}°, Best Intensity = {final_env.best_intensity:.3f}")
    hapu_phase_module.stop_output()
    
# 最後のリセット
print(phase)
# speakerON(9)
