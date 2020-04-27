import numpy as np
import torch
import datetime
from unityagents import UnityEnvironment
# from ddqn_agent import MultiAgent, Agent
from ddqn_agent import Agent
from collections import deque

import argparse

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay
# WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1
UPDATE_TIMES = 1

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--filename', type=str, default='app/Reacher.app', help='the application you would like to run')
parser.add_argument('--buffer_size', type=int, default=BUFFER_SIZE, help='size of the buffer')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='the size of batches to be processed')
parser.add_argument('--lr_actor', type=float, default=LR_ACTOR, help='actors learning rate')
parser.add_argument('--lr_critic', type=float, default=LR_CRITIC, help='critics learning rate')
parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='weigth decay in the critics optimizer')
parser.add_argument('--update_every', type=int, default=UPDATE_EVERY, help='how often to update the network')
parser.add_argument('--update_times', type=int, default=UPDATE_TIMES, help='how many times to update the network each update step')
parser.add_argument('--seed', type=int, default=2, help='random seed value')
parser.add_argument('--gamma', type=float, default=GAMMA, help='gamma value')
args = parser.parse_args()

# env = UnityEnvironment(file_name='app/Reacher.app')
# env = UnityEnvironment(file_name='app/ReacherSingle.app')
env = UnityEnvironment(file_name=args.filename)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# agent = MultiAgent(state_size=33, action_size=4, num_agents=20, random_seed=2)
# agent = MultiAgent(state_size=33, action_size=4, num_agents=1, random_seed=2)
agent = Agent(
    state_size=33,
    action_size=4,
    random_seed=args.seed,
    num_agents=num_agents,
    buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    lr_actor=args.lr_actor,
    lr_critic=args.lr_critic,
    weight_decay=args.weight_decay,
    update_every=args.update_every,
    update_times=args.update_times,
    gamma=args.gamma)

import matplotlib.pyplot as plt
# %matplotlib inline

def ddpg(n_episodes=1000, max_t=1200, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    times = []
    for i_episode in range(1, n_episodes+1):
        ep_scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        time_a = datetime.datetime.now()
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            ep_scores += env_info.rewards

            # print('\rEpisode {}\tScore: {}\tTimestep: {}\t\t'.format(i_episode, ep_scores.mean(), t), end="")

            # need to step for each state
            # for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            #     agent.step(state, action, reward, next_state, done)
            agent.step(states, actions, rewards, next_states, dones)

            states = next_states

            if np.any(dones):
                break

        time_b = datetime.datetime.now()
        time = time_b - time_a
        times.append(time_b - time_a)
        time_average = np.mean(times)
        time_remaining = (n_episodes*time_average.total_seconds()-i_episode*time_average.total_seconds())/60/60

        scores_deque.append(ep_scores)
        scores.append(ep_scores)

        hours = int(time_remaining)
        minutes = int((time_remaining*60) % 60)
        seconds = int((time_remaining*3600) % 60)

        print('Episode {}\tAverage Score: {:.2f}\tAverage Time Per Episode: {}\tTime to Complete: {:02d}:{:02d}:{:02d}'.format(
            i_episode,
            np.mean(scores_deque),
            time_average,
            hours,
            minutes,
            seconds
        ), end="\n")

        agent.save('checkpoint_actor_ddqn', 'checkpoint_critic_ddqn')
        # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) > 30:
            print('\nSolved! Episode: {}, Solved after {} episodes! Average score over last 100 episodes: {}'.format(
                i_episode, i_episode - 100, np.mean(scores_deque)
            ))
            break

    return scores

# test execution
# scores = ddpg(n_episodes=2, max_t=1200, print_every=100)
scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png')
# plt.show()