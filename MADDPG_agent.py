import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, num_agents, state_size, action_size, random_seed,buffer_size, batch_size,gamma, TAU,
                 lr_actor, lr_critic, weight_decay, a_hidden_sizes, c_hidden_sizes):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # Hyperparameters
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay
        self.ACTOR_HL_SIZE= a_hidden_sizes               
        self.CRITIC_HL_SIZE= c_hidden_sizes 
        self.num_agents = num_agents
        

        # Actor Network (w/ Target Network)
        self.actor_local_1 = Actor(state_size, action_size, random_seed,self.ACTOR_HL_SIZE).to(device)
        self.actor_target_1 = Actor(state_size, action_size, random_seed,self.ACTOR_HL_SIZE).to(device)
        self.actor_optimizer_1 = optim.Adam(self.actor_local_1.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local_1 = Critic(state_size, action_size, random_seed, self.CRITIC_HL_SIZE).to(device)
        self.critic_target_1 = Critic(state_size, action_size, random_seed, self.CRITIC_HL_SIZE).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=self.LR_CRITIC, weight_decay= self.WEIGHT_DECAY)
        
        # Actor Network (w/ Target Network)
        self.actor_local_2 = Actor(state_size, action_size, random_seed,self.ACTOR_HL_SIZE).to(device)
        self.actor_target_2 = Actor(state_size, action_size, random_seed,self.ACTOR_HL_SIZE).to(device)
        self.actor_optimizer_2 = optim.Adam(self.actor_local_2.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local_2 = Critic(state_size, action_size, random_seed, self.CRITIC_HL_SIZE).to(device)
        self.critic_target_2 = Critic(state_size, action_size, random_seed, self.CRITIC_HL_SIZE).to(device)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=self.LR_CRITIC, weight_decay= self.WEIGHT_DECAY)


        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
        
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(states.shape[0]):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA)    

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local_1.eval()
        self.actor_local_2.eval()
        action_values = [states.shape[0],self.action_size]
        with torch.no_grad():
            action_values[0] = self.actor_local_1(states[0]).cpu().data.numpy()
            action_values[1] = self.actor_local_2(states[1]).cpu().data.numpy()
        self.actor_local_1.train()
        self.actor_local_2.train()
        
        #print (action_values)
        if add_noise:
            action_values += self.noise.sample()
        #print (action_values)
        #print (np.clip(action_values, -1, 1))
        return np.clip(action_values, -1, 1)


    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next_1 = self.actor_target_1(next_states)
        actions_next_2 = self.actor_target_2(next_states)
        Q_targets_next_1 = self.critic_target_1(next_states, actions_next_1.detach())
        Q_targets_next_2 = self.critic_target_2(next_states, actions_next_2.detach())
        # Compute Q targets for current states (y_i)
        Q_targets_1 = rewards + (gamma * Q_targets_next_1 * (1 - dones))
        Q_targets_2 = rewards + (gamma * Q_targets_next_2 * (1 - dones))
        # Compute critic loss
        Q_expected_1 = self.critic_local_1(states, actions)
        Q_expected_2 = self.critic_local_2(states, actions)
        critic_loss_1 = F.mse_loss(Q_expected_1, Q_targets_1.detach())
        critic_loss_2 = F.mse_loss(Q_expected_2, Q_targets_2.detach())
        # Minimize the loss
        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # adds gradient clipping to stabilize learning
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred_1 = self.actor_local_1(states)
        actions_pred_2 = self.actor_local_2(states)
        actor_loss_1 = -self.critic_local_1(states, actions_pred_1).mean()
        actor_loss_2 = -self.critic_local_2(states, actions_pred_2).mean()
        # Minimize the loss
        self.actor_optimizer_1.zero_grad()
        self.actor_optimizer_2.zero_grad()
        actor_loss_1.backward()
        actor_loss_2.backward()
        self.actor_optimizer_1.step()
        self.actor_optimizer_2.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local_1, self.critic_target_1, self.TAU)
        self.soft_update(self.critic_local_2, self.critic_target_2, self.TAU)
        self.soft_update(self.actor_local_1, self.actor_target_1, self.TAU)
        self.soft_update(self.actor_local_2, self.actor_target_2, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size 
        self.reset()
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)