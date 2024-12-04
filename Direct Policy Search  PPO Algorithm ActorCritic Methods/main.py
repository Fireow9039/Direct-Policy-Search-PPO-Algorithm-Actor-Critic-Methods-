import gym
import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from Modules import ActorNetwork, CriticNetwork
from torch_misc import gather_trajectories, calculate_gae, apply_ppo_update, ExperienceBuffer

# Initialize environment
environment = gym.make('Pendulum-v1')
observation_dim = environment.observation_space.shape[0]
action_dim = environment.action_space.shape[0]

# Training parameters
num_epochs = 30
steps_per_cycle = 4000
discount_factor = 0.99
gae_lambda = 0.95
ppo_clip = 0.2
actor_lr = 3e-4
critic_lr = 1e-3
hidden_layers = [64, 64]
buffer_limit = 10000
mini_batch_size = 64

# Initialize networks and optimizers
actor = ActorNetwork(observation_dim, action_dim, hidden_layers)
critic = CriticNetwork(observation_dim, hidden_layers)
actor_optimizer = Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = Adam(critic.parameters(), lr=critic_lr)
experience_buffer = ExperienceBuffer(buffer_limit)

# Metrics storage
reward_tracking = {}
policy_loss_tracking = {}
critic_loss_tracking = {}

# Experiment configurations
experiments = {
    "clipping_enabled": (True, True),
    "clipping_disabled": (False, True),
    "gae_enabled": (True, True),
    "gae_disabled": (True, False),
}

# Training loop for all experiments
for experiment_name, (enable_clipping, enable_gae) in experiments.items():
    rewards = []
    policy_losses = []
    value_losses = []

    for cycle in range(num_epochs):
        # Collect data from environment interaction
        sampled_data = gather_trajectories(environment, actor, steps_per_cycle)

        # Store data in experience buffer
        for record in sampled_data:
            experience_buffer.add(record)

        # Sample a batch from buffer
        data_batch = experience_buffer.retrieve(mini_batch_size)
        states, actions, advantage_estimates, discounted_returns, prev_log_probs = calculate_gae(
            data_batch, critic, discount_factor, gae_lambda if enable_gae else 0.0
        )

        # Perform PPO update
        actor_loss, critic_loss = apply_ppo_update(
            actor, critic, actor_optimizer, critic_optimizer, states, actions,
            advantage_estimates, discounted_returns, prev_log_probs, 
            ppo_clip if enable_clipping else float('inf')
        )

        # Calculate mean rewards
        average_reward = np.mean([record[2] for record in sampled_data])

        # Track metrics
        rewards.append(average_reward)
        policy_losses.append(actor_loss)
        value_losses.append(critic_loss)

        print(f"{experiment_name} - Cycle {cycle + 1}: Reward: {average_reward:.2f}")

    # Save results
    reward_tracking[experiment_name] = rewards
    policy_loss_tracking[experiment_name] = policy_losses
    critic_loss_tracking[experiment_name] = value_losses

# Plot results
plt.figure(figsize=(16, 10))

# Reward plots
plt.subplot(2, 2, 1)
for exp_name, rewards in reward_tracking.items():
    plt.plot(range(1, num_epochs + 1), rewards, label=exp_name)
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.title("Learning Curves")
plt.legend()

# Policy loss plots
plt.subplot(2, 2, 2)
for exp_name, losses in policy_loss_tracking.items():
    plt.plot(range(1, num_epochs + 1), losses, label=exp_name)
plt.xlabel("Epoch")
plt.ylabel("Policy Loss")
plt.title("Policy Loss Curves")
plt.legend()

# Critic loss plots (log scale)
plt.subplot(2, 2, 3)
for exp_name, losses in critic_loss_tracking.items():
    scaled_loss = np.log(np.array(losses) + 1e-5)
    plt.plot(range(1, num_epochs + 1), scaled_loss, label=exp_name)
plt.xlabel("Epoch")
plt.ylabel("Log(Value Loss)")
plt.title("Value Loss Curves")
plt.legend()

plt.tight_layout()
plt.show()

# 2D Value Landscape Visualization
theta_vals = np.linspace(-np.pi, np.pi, 100)
theta_dot_vals = np.linspace(-8, 8, 100)
V_vals = np.zeros((100, 100))

# Evaluate value function over a grid of states
for i, theta in enumerate(theta_vals):
    for j, theta_dot in enumerate(theta_dot_vals):
        state = np.array([np.sin(theta), np.cos(theta), theta_dot])
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        V_vals[j, i] = critic(state_tensor).item()

# Plot the value landscape
plt.figure(figsize=(6, 5))
plt.imshow(V_vals, extent=[-np.pi, np.pi, -8, 8], origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="V(s)")
plt.xlabel("Theta")
plt.ylabel("Theta Dot")
plt.title("2D Landscape of V(s)")
plt.show()
