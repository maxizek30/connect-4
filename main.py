from environment import Connect4Env
from replay_buffer import ReplayBuffer
from model import create_model
from training import train_step
import numpy as np
import tensorflow as tf
import math

# Check for GPU availability
print("GPUs available:", len(tf.config.list_physical_devices('GPU')))

# Initialize environment and replay buffer
env = Connect4Env()
replay_buffer = ReplayBuffer(max_size=1_000_000)

# Define the state and action spaces
state_shape = (6, 7) # Board dimensions
num_actions = 7 # Number of columns (actions) in Connect4

# Create the Q-network and the target network
model = create_model(state_shape, num_actions) # Q-network
target_model = create_model(state_shape, num_actions) # Target network
# Initialize target model weights to match the Q-network
target_model.set_weights(model.get_weights())

# Hyperparameters
batch_size = 128 # Number of samples per training batch
gamma = 0.95  # Discount factor for future rewards
epsilon = 1.0 # Initial exploration rate (probability of random actions)
epsilon_min = 0.1 # Minimum exploration rate
epsilon_decay = 0.99995  # Rate at which epsilon decays after each episode
target_update_freq = 10 # Frequency (in episodes) to update the target network


def evaluate_agent(env, model, num_games=10):
    """
    Evaluate the agent's performance over a specified number of games.

    Args:
    - env: The Connect4 game environment.
    - model: The trained Q-network model.
    - num_games: Number of games to play for evaluation.

    Prints the win rate of the agent.
    """
    wins = 0
    for _ in range(num_games):
        state = env.reset() # Reset environment for a new game
        state = state[..., np.newaxis]  # Add channel dimension for model input
        done = False
        while not done:
            valid_cols = env.valid_actions() # Get valid actions (columns)
            q_values = model.predict(state[np.newaxis, ...])[0] # Predict Q-values
            # Select the best valid column
            best_valid_col = valid_cols[np.argmax([q_values[c] for c in valid_cols])]
            state, _, done = env.step(best_valid_col) # Take the action
            state = state[..., np.newaxis]  # Add channel dimension
        # Check if player 1 wins
        if env.current_player == -1:  # Opponent lost
            wins += 1
    print(f"Win Rate: {wins / num_games:.2f}") # Print the win rate

def evaluate_agent_selfplay(env, model, num_games=10):
    """
    Evaluate the agent by playing self-play games:
    - Player 1 uses the same model.
    - Player 2 uses the same model.
    The agent effectively plays against itself.

    Args:
        env: The Connect4Env environment.
        model: The trained Q-network to evaluate.
        num_games (int): Number of self-play games to simulate.

    Prints the number of wins for Player 1, wins for Player 2, and any ties.
    """
    wins_p1 = 0
    wins_p2 = 0

    for _ in range(num_games):
        state = env.reset()  # reset board
        done = False

        while not done:
            # === Player 1 move ===
            valid_cols = env.valid_actions()

            # Expand dimensions for the model: (1, rows, cols, 1)
            state_input = state[..., np.newaxis][np.newaxis, ...]
            q_values = model.predict(state_input)[0]

            # Among valid columns, pick the one with highest Q-value
            best_valid_col = valid_cols[np.argmax([q_values[c] for c in valid_cols])]
            state, reward, done = env.step(best_valid_col)

            if done:
                # After env.step(), the environment flips current_player.
                # The *winner* is the one who just played => -env.current_player
                winner = -1 * env.current_player
                if winner == 1:
                    wins_p1 += 1
                elif winner == -1:
                    wins_p2 += 1
                break

            # === Player 2 move ===
            valid_cols = env.valid_actions()
            state_input = state[..., np.newaxis][np.newaxis, ...]
            q_values = model.predict(state_input)[0]
            best_valid_col = valid_cols[np.argmax([q_values[c] for c in valid_cols])]
            state, reward, done = env.step(best_valid_col)

            if done:
                winner = -1 * env.current_player
                if winner == 1:
                    wins_p1 += 1
                elif winner == -1:
                    wins_p2 += 1
                break

    ties = num_games - wins_p1 - wins_p2
    print(f"Self-Play over {num_games} games => P1 wins: {wins_p1}, P2 wins: {wins_p2}, Ties: {ties}")


# Training loop
episode = 0
while True:  # runs forever until Ctrl-C
    # Save the model every 50 episodes
    if episode % 1000 == 0:
        # Use the TensorFlow SavedModel format (directory-based)
        model.save(f"models/connect4_model_episode_{episode}", save_format="tf")
    # Evaluate the agent every 10 episodes
    if episode % 1000 == 0:
        evaluate_agent_selfplay(env, model, num_games=50)


    # Reset the environment for a new episode
    state = env.reset()
    state = state[..., np.newaxis]  # Add channel dimension

    done = False
    total_reward = 0 # Initialize total reward for the episode

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Exploration: pick randomly among valid columns
            action = np.random.choice(env.valid_actions())
        else:
            # Exploitation: pick the column with the highest Q among valid columns
            valid_cols = env.valid_actions()
            q_values = model.predict(state[np.newaxis, ...])[0]
            # Get the Q-values only for valid columns, then pick whichever is max
            best_valid_col = valid_cols[np.argmax([q_values[c] for c in valid_cols])]
            action = best_valid_col

        # Take the action and observe the result
        next_state, reward, done = env.step(action)
        next_state = next_state[..., np.newaxis]  # Add channel dimension
        total_reward += reward
        # Store the experience in the replay buffer
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state # Update the current state

        # Perform a training step
        train_step(model, target_model, replay_buffer, batch_size, gamma)

    # Decay epsilon (exploration rate) after each episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update the target network periodically
    if episode % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    # Print episode details
    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    episode += 1







