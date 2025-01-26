def train_step(model, target_model, replay_buffer, batch_size, gamma):
    if replay_buffer.size() < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    # Reshape states and next_states to include the channel dimension
    states = states[..., np.newaxis]  # Add channel dimension
    next_states = next_states[..., np.newaxis]  # Add channel dimension

    # Predict Q-values
    q_values = model.predict(states)
    next_q_values = target_model.predict(next_states)

    target_q_values = q_values.copy()
    for i in range(batch_size):
        if dones[i]:
            target_q_values[i, actions[i]] = rewards[i]
        else:
            target_q_values[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])

    model.train_on_batch(states, target_q_values)
