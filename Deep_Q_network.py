import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# -------------------------------
# 1. Define DQN Network
# -------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# 2. Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# -------------------------------
# 3. Training Parameters
# -------------------------------
state_dim = 4  # e.g., x, y, vx, vy (modify according to your ASV state)
action_dim = 5 # e.g., forward, backward, left, right, stay
lr = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
batch_size = 64
buffer_capacity = 10000
num_episodes = 500

# -------------------------------
# 4. Initialize
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(buffer_capacity)

# -------------------------------
# 5. Example ASV environment (placeholder)
# -------------------------------
class ASVEnv:
    def reset(self):
        # Return initial state (x, y, vx, vy)
        return np.zeros(state_dim, dtype=np.float32)
    
    def step(self, action):
        # Apply action to ASV dynamics and return next_state, reward, done
        next_state = np.random.randn(state_dim).astype(np.float32)
        reward = random.random()
        done = random.random() < 0.05
        return next_state, reward, done, {}

env = ASVEnv()

# -------------------------------
# 6. Training Loop
# -------------------------------
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, action_dim-1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax().item()
        
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # Sample mini-batch and train
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
            
            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + gamma * max_next_q * (1 - dones)
            
            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Update target network periodically
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

print("Training Complete!")
