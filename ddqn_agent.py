import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

###################### DEAFULTS ######################
# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 128        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
# LR_ACTOR = 1e-4         # learning rate of the actor
# LR_CRITIC = 1e-3        # learning rate of the critic
# WEIGHT_DECAY = 0        # L2 weight decay
###################### DEAFULTS ######################

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
# WEIGHT_DECAY = 0.0001   # L2 weight decay
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1
UPDATE_TIMES = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("training on GPU")
else:
    print("training on CPU")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 random_seed,
                 num_agents = 1,
                 buffer_size = BUFFER_SIZE,
                 batch_size = BATCH_SIZE,
                 gamma = GAMMA,
                 tau = TAU,
                 lr_actor = LR_ACTOR,
                 lr_critic = LR_CRITIC,
                 weight_decay = WEIGHT_DECAY,
                 update_every = UPDATE_EVERY,
                 update_times = UPDATE_TIMES):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            with_memory (boolean): shared memory in another agent
            with_critic (boolean): initialize the critic
        """

        self.num_agents = num_agents

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.update_every = update_every
        self.update_times = update_times

        print("BUFFER_SIZE: {}".format(self.buffer_size))
        print("BATCH_SIZE: {}".format(self.batch_size))
        print("GAMMA: {}".format(self.gamma))
        print("TAU: {}".format(self.tau))
        print("LR_ACTOR: {}".format(self.lr_actor))
        print("LR_CRITIC: {}".format(self.lr_critic))
        print("WEIGHT_DECAY: {}".format(self.weight_decay))
        print("UPDATE_EVERY: {}".format(self.update_every))
        print("UPDATE_TIMES: {}".format(self.update_times))

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actors_local = []
        self.actors_target = []
        self.actors_optimizer = []

        self.critics_local = []
        self.critics_target = []
        self.critics_optimizer = []

        for i in range(self.num_agents):
            # Actor Network (w/ Target Network)
            self.actors_local.append(Actor(state_size, action_size, random_seed).to(device))
            self.actors_target.append(Actor(state_size, action_size, random_seed).to(device))
            self.actors_optimizer.append(optim.Adam(self.actors_local[i].parameters(), lr=self.lr_actor))

            # Critic Network (w/ Target Network)
            self.critics_local.append(Critic(state_size, action_size, random_seed).to(device))
            self.critics_target.append(Critic(state_size, action_size, random_seed).to(device))
            self.critics_optimizer.append(optim.Adam(self.critics_local[i].parameters(), lr=self.lr_critic, weight_decay=self.weight_decay))

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.t_step = 0

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every

        if len(self.memory) > self.batch_size and self.t_step == 0:
            for _ in range(self.update_times):
                for i in range(self.num_agents):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma, i)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = []
        for i in range(self.num_agents):
            agent_state = torch.from_numpy(state[i]).float().to(device)
            self.actors_local[i].eval()
            with torch.no_grad():
                action = self.actors_local[i](agent_state).cpu().data.numpy()
            self.actors_local[i].train()
            if add_noise:
                action += self.noise.sample()
            actions.append(np.clip(action, -1, 1))
        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_index):
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
        actions_next = self.actors_target[agent_index](next_states)
        Q_targets_next = self.critics_target[agent_index](next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critics_local[agent_index](states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critics_optimizer[agent_index].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics_local[agent_index].parameters(), 1)
        self.critics_optimizer[agent_index].step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actors_local[agent_index](states)
        actor_loss = -self.critics_local[agent_index](states, actions_pred).mean()
        # Minimize the loss
        self.actors_optimizer[agent_index].zero_grad()
        actor_loss.backward()
        self.actors_optimizer[agent_index].step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critics_local[agent_index], self.critics_target[agent_index], self.tau)
        self.soft_update(self.actors_local[agent_index], self.actors_target[agent_index], self.tau)

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

    def save(self, actor, critic):
        for i in range(self.num_agents):
            torch.save(self.actors_local[i].state_dict(), 'checkpoints/{}_{}.pth'.format(actor, i))
            torch.save(self.critics_local[i].state_dict(), 'checkpoints/{}_{}.pth'.format(critic, i))

    def load(self, actor):
        for i in range(self.num_agents):
            self.actors_local[i].load_state_dict(torch.load('checkpoints/{}_{}.pth'.format(actor, i), map_location='cpu'))

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1): # seems all over the place
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
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