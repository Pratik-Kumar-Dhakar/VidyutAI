import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import os
from tqdm import tqdm

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


## 1. BUILDING ENVIRONMENT
class BuildingEnv:
    """
    Simulates the building's energy system (PV, ESS, Grid).
    The environment provides states and rewards to the RL agent.
    """
    def __init__(self, data):
        self.data = data
        self.ess_capacity_kwh = 1000.0
        self.max_power_kw = self.ess_capacity_kwh  # Assume can be charged/discharged in 1 hour
        self.current_time_step = 0
        self.ess_soc = 0.5  # Initial SOC is 0.5 (50%)

        # Action space: 11 discrete levels of charge/discharge rate
        self.action_space = np.linspace(-1.0, 1.0, 11)

    def _get_state(self):
        """
        Constructs the state vector from the current time step's data.
        State: [PV, Load, Sell Price, Buy Price, SOC, Avg Sell Price, Peak Load]
        """
        row = self.data.iloc[self.current_time_step]

        # Set buying price based on current time step
        if self.current_time_step < 6:
            buying_price_per_mwh = 2.5 * 1000
        elif self.current_time_step < 18:
            buying_price_per_mwh = 4 * 1000
        else:
            buying_price_per_mwh = 7.5 * 1000

        avg_selling_price = 3 * 1000
        peak_load = self.data['New demand'].max()

        state = np.array([
            row['Wind+Solar'],
            row['New demand'],
            3.0 * 1000,
            buying_price_per_mwh,
            self.ess_soc,
            avg_selling_price,
            peak_load
        ])
        return state

    def reset(self):
        """Resets the environment for a new episode."""
        self.current_time_step = 0
        self.ess_soc = 0.0  # Initial SOC is 0
        return self._get_state()

    def step(self, action_index):
        """
        Executes one time step in the environment.
        1. Takes an action.
        2. Updates the ESS State of Charge (SOC).
        3. Calculates the monetary reward/cost for the hour.
        4. Returns the next state, reward, and done flag.
        """
        action = self.action_space[action_index]

        current_data = self.data.iloc[self.current_time_step]
        pv_gen = current_data['Wind+Solar']
        load = current_data['New demand']

        if self.current_time_step < 6:
            cp = 2.5 * 1000
            sp = 3 * 1000
        elif self.current_time_step < 18:
            cp = 4 * 1000
            sp = 3 * 1000
        else:
            cp = 7.5 * 1000
            sp = 3 * 1000

        # The desired energy change in kWh for this hour
        desired_energy_change = action * self.max_power_kw

        net_load = load - pv_gen

        # Implement physical constraints
        # Can only charge from surplus PV, can't discharge more than available SOC
        if desired_energy_change > 0:  # Charging
            chargeable_energy = max(0, pv_gen - load)
            energy_change = min(desired_energy_change, chargeable_energy, (1 - self.ess_soc) * self.ess_capacity_kwh)
        else:  # Discharging
            energy_change = -min(abs(desired_energy_change), self.ess_soc * self.ess_capacity_kwh)

        # Update SOC
        self.ess_soc += energy_change / self.ess_capacity_kwh

        # Calculate reward (monetary benefit)
        reward = 0
        if energy_change < 0:  # Discharging
            benefit_price = max(cp, sp)
            reward = abs(energy_change) * benefit_price

        # Move to next time step
        self.current_time_step += 1
        done = self.current_time_step >= len(self.data) - 1

        next_state = self._get_state() if not done else np.zeros_like(self._get_state())

        return next_state, reward, done


## 2. DQN AGENT
class DQNAgent:
    """Deep Q-Network Agent"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory

        # Hyperparameters
        self.gamma = 0.95      # discount rate
        self.epsilon = 1.0     # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the Neural Network for Q-value function approximation.
        """
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selects an action using epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Trains the network using a random minibatch from replay memory in batch.
        """
        minibatch = random.sample(self.memory, batch_size)

        states = np.vstack([e[0] for e in minibatch])
        actions = [e[1] for e in minibatch]
        rewards = [e[2] for e in minibatch]
        next_states = np.vstack([e[3] for e in minibatch])
        dones = [e[4] for e in minibatch]

        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        self.model.fit(states, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


## 3. MAIN TRAINING LOOP
if __name__ == "__main__":

    # === GPU DETECTION ===
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✅ Found {len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs. Training will use the GPU. 🚀")
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ No GPU found. TensorFlow will use the CPU. Training may be slow.")

    # --- LOAD DATA ---
    try:
        data_df = pd.read_excel('my_energy_data.xlsx')
        print("Successfully loaded 'my_energy_data.xlsx'.")
    except FileNotFoundError:
        print("\nERROR: 'my_energy_data.xlsx' not found.")
        print("Please create this file with the required columns and run the script again.")
        exit()

    # Initialize environment and agent
    env = BuildingEnv(data_df)
    state_size = env._get_state().shape[0]
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)

    episodes = 20
    batch_size = 40
    train_every = 5  # Train every 5 time steps

    print(f"\nStarting training for {episodes} episodes...")
    print(f"State Size: {state_size}, Action Size: {action_size}")

    episode_pbar = tqdm(range(episodes), desc="Training Progress")

    for e in episode_pbar:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(len(data_df)):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

            # Train every few steps, not every step
            if time % train_every == 0 and len(agent.memory) > batch_size:
                agent.replay(batch_size)

        episode_pbar.set_postfix(
            {
                "Total Reward": f"${total_reward:,.2f}",
                "Epsilon": f"{agent.epsilon:.2f}"
            }
        )

    print("\n\nTraining finished.")
# Save the trained model after all episodes
agent.model.save("dqn_model.h5")
print("Model saved successfully to 'dqn_model.h5'")
