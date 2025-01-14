from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import random
import pickle

# Initialize the environment
print("Initializing the environment...")
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
print("Environment initialized.")

# Define the agent class
class ProjectAgent:
    def __init__(self, action_space=4, epsilon=0.1):
        """
        Initialize the agent with a simple epsilon-greedy policy.
        """
        self.action_space = action_space
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Placeholder for a Q-table or any model
        print("Agent initialized with epsilon:", epsilon)

    def act(self, observation, use_random=False):
        """
        Decide the action based on the observation.
        If use_random is True or with probability epsilon, take a random action.
        Otherwise, take a fixed action (e.g., action 0).
        """
        if use_random or random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space))
        return 0  # Fixed action for simplicity

    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's policy or knowledge.
        Placeholder implementation for a Q-table.
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0] * self.action_space
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0] * self.action_space

        # Simple Q-learning update rule
        alpha = 0.1  # Learning rate
        gamma = 0.99  # Discount factor

        best_next_action = max(self.q_table[next_state_key])
        target = reward + (gamma * best_next_action * (not done))
        self.q_table[state_key][action] += alpha * (target - self.q_table[state_key][action])

    def save(self, path):
        """
        Save the agent's state or model to a file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Agent state saved to {path}.")

    def load(self, path="agent_model.pkl"):
        """
        Load the agent's state or model from a file.
        """
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Agent state loaded from {path}.")

# Instantiate the agent with 4 actions (corresponding to the HIV environment)
print("Instantiating the agent...")
agent = ProjectAgent(action_space=4)
print("Agent instantiated.")

# Training loop
num_episodes = 3
print("Starting training...")
for episode in range(num_episodes):
    print(f"Episode {episode + 1}/{num_episodes} in progress...")
    state, _ = env.reset()
    done = False
    total_reward = 0
    step_counter = 0  # Step counter to track the number of steps

    while not done:
        step_counter += 1
        if step_counter > 200:  # Safety check to prevent infinite loops
            print(f"Exceeded 200 steps in Episode {episode + 1}. Forcing exit.")
            break

        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if step_counter % 50 == 0:  # Print progress every 50 steps
            print(f"Step {step_counter}: Total Reward = {total_reward}")

    print(f"Episode {episode + 1} complete. Total Reward = {total_reward}, Steps = {step_counter}")

# Save the trained agent
agent.save("agent_model.pkl")

print("Training complete.")
# Close the environment
env.close()
print("Environment closed.")
