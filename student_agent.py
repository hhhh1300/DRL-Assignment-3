import gym
import os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from collections import deque
import torch.nn as nn
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

        
def orthogonal_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
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


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts based on a trained DQN model."""
    def __init__(self):
        self.n_frames = 4 
        self.action_space = gym.spaces.Discrete(12)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        # torch.set_num_threads(os.cpu_count())

        self.qnet = DuelingQNet(self.n_frames, self.action_space.n).to(self.device)
        qnet_model_path = "./model.pth" 
        self.load_model(qnet_model_path)
        self.frames = deque(maxlen=self.n_frames) 
        self.needs_reset = True
        self.frame_skip = 4     
        self.steps_since_last_decision = 0 
        self.last_action = 0    
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),                               
            transforms.Grayscale(num_output_channels=1),           
            transforms.Resize((84, 84), interpolation=InterpolationMode.BOX),
            transforms.PILToTensor()                               
        ])


    def load_model(self, qnet_path):
        """Loads the Q-network model weights."""
        if os.path.exists(qnet_path):
            try:
                self.qnet.load_state_dict(torch.load(qnet_path, weights_only=True, map_location=self.device))
                print(f"QNet loaded from {qnet_path}")
                self.qnet.eval() 
            except Exception as e:
                 print(f"Error loading QNet model from {qnet_path}: {e}")
                 raise 
        else:
             print(f"Error: QNet model file not found at {qnet_path}")
             raise FileNotFoundError(f"Model file not found: {qnet_path}")

    def reset(self):
        """Resets the agent's internal state (frame buffer)."""
        self.frames.clear()
        self.needs_reset = True
        self.steps_since_last_decision = 0
        self.last_action = 0 

    def _preprocess_observation(self, observation):
        """Converts raw observation (RGB) to processed frame (Gray, Resized)."""
        frame = self.transform(observation)    
        return frame.squeeze(0).numpy() 

    def act(self, observation):
        """
        Processes the observation, updates the frame stack, and returns an action
        based on the DQN model, handling frame skipping.
        """
        self.steps_since_last_decision += 1
        if not self.needs_reset and self.steps_since_last_decision % self.frame_skip != 1:
             return self.last_action

        processed_frame = self._preprocess_observation(observation)

        if self.needs_reset:
            for _ in range(self.n_frames):
                self.frames.append(processed_frame)
            self.needs_reset = False
            self.steps_since_last_decision = 1 
        else:
            self.frames.append(processed_frame)

        state_np = np.stack(self.frames, axis=0) # Shape: (4, 84, 84) uint8
        state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            q_values = self.qnet(state_tensor) # Pass normalized tensor
        action = torch.argmax(q_values, dim=1).item()
        self.last_action = action # Store the decided action
        return action

if __name__ == "__main__":
    print("Initializing environment...")
    env = gym_super_mario_bros.make('SuperMarioBros-v0') # Or your target level e.g. 'SuperMarioBros-1-1-v0'
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    print("Environment initialized.")

    print("Initializing agent...")
    agent = Agent()
    print(f"Agent initialized using device: {agent.device}")

    print("Starting evaluation...")
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    # add random action to the first four frames
    for _ in range(4):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

    try:
        while True: # Loop for multiple episodes if needed, or just one run
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            # print (f"Step: {step_count}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")
            if done:
                print("-" * 20)
                print(f"Episode finished.")
                print(f"Total reward: {total_reward}")
                print(f"Steps taken: {step_count}")
                print(f"Mario Status: {info.get('flag_get', 'Did not finish')}") # Check if flag reached
                print("-" * 20)

                # Reset for next episode or break
                obs = env.reset()
                agent.reset()     # Reset agent's frame buffer
                total_reward = 0
                step_count = 0
                # Uncomment the break if you only want to run one episode
                break

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        print("Closing environment.")
        env.close()