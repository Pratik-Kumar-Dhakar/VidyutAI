import numpy as np
import tensorflow as tf

def load_model(path="dqn_model.h5"):
    """
    Load the trained DQN model without compiling to avoid deserialization errors.
    """
    model = tf.keras.models.load_model(path, compile=False)
    return model

def build_state(wind_solar, load, soc, current_time_step, peak_load, avg_selling_price=3000):
    """
    Constructs the state vector exactly as in your environment.
    """

    # Buying price depends on time of day
    if current_time_step < 6:
        buying_price = 2.5 * 1000
    elif current_time_step < 18:
        buying_price = 4 * 1000
    else:
        buying_price = 7.5 * 1000

    sell_price = 3 * 1000

    state = np.array([
        wind_solar,
        load,
        sell_price,
        buying_price,
        soc,
        avg_selling_price,
        peak_load
    ])

    return np.reshape(state, [1, len(state)])

def predict_action(model, state):
    """
    Given a model and state, predict the best action index and action value.
    """
    q_values = model.predict(state, verbose=0)
    action_index = np.argmax(q_values[0])

    # Same action space as training:
    action_space = np.linspace(-1.0, 1.0, 11)
    selected_action = action_space[action_index]

    return action_index, selected_action

def main():
    # Load the trained model
    model = load_model("dqn_model.h5")

    # Example input (you can replace these with your actual values)
    wind_solar = 1000       # PV generation in kW
    load = 200             # Load in kW
    soc = 0.1              # State of Charge (0 to 1)
    current_time_step = 10  # Hour of the day or timestep
    peak_load = 300

    # Build state vector
    state = build_state(wind_solar, load, soc, current_time_step, peak_load)

    # Predict best action
    action_index, action_value = predict_action(model, state)

    print(f"Predicted action index: {action_index}")
    print(f"Corresponding charge/discharge level: {action_value}")

if __name__ == "__main__":
    main()
