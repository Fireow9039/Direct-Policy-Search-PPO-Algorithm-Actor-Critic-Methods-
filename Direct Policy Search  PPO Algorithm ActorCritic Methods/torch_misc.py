import numpy as np
import torch
from collections import deque
from torch.distributions import Normal


# Replay buffer to store and retrieve experiences
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, data):
        self.buffer.append(data)

    def retrieve(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

# Collect experiences from environment
def gather_trajectories(env, policy, steps):
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    collected_data = []

    for _ in range(steps):
        obs_tensor = torch.as_tensor(np.array(observation), dtype=torch.float32)
        action, log_prob = policy.sample_action(obs_tensor)
        next_obs, reward, done, truncated, _ = env.step(action)

        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]

        collected_data.append((observation, action, reward, log_prob))

        if done or truncated:
            observation = env.reset()[0]
        else:
            observation = next_obs

    return collected_data

# Calculate Generalized Advantage Estimation (GAE)
def calculate_gae(trajectory, value_fn, discount_factor, lambda_gae):
    obs, acts, rewards, log_probs = zip(*trajectory)
    obs = torch.as_tensor(np.array(obs), dtype=torch.float32)
    acts = torch.as_tensor(np.array(acts), dtype=torch.float32)
    rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32)
    log_probs = torch.as_tensor(np.array(log_probs), dtype=torch.float32)

    values = value_fn(obs).detach()
    deltas = rewards[:-1] + discount_factor * values[1:] - values[:-1]
    advantages = torch.zeros_like(rewards)
    adv = 0.0

    for t in reversed(range(len(deltas))):
        adv = deltas[t] + discount_factor * lambda_gae * adv
        advantages[t] = adv

    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    returns = normalized_rewards + discount_factor * advantages
    return obs, acts, advantages, returns, log_probs

# Apply Proximal Policy Optimization (PPO) updates
def apply_ppo_update(actor, critic, actor_opt, critic_opt, obs, acts, advantages,
                     returns, old_log_probs, clip_threshold):
    avg_actor_loss = 0
    avg_critic_loss = 0

    for _ in range(10):
        # Update actor
        actor_opt.zero_grad()
        mean, std = actor(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(acts).sum(axis=-1)
        ratios = torch.exp(log_probs - old_log_probs)
        clipped_ratios = torch.clamp(ratios, 1 - clip_threshold, 1 + clip_threshold)
        actor_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
        actor_loss.backward()
        actor_opt.step()
        avg_actor_loss += actor_loss.item()

        # Update critic
        critic_opt.zero_grad()
        critic_loss = ((critic(obs) - returns) ** 2).mean()
        critic_loss.backward()
        critic_opt.step()
        avg_critic_loss += critic_loss.item()

    return avg_actor_loss / 10, avg_critic_loss / 10
