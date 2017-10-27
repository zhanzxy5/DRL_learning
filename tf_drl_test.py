import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import gym

# Simple CartPole with naive policy
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

def drl_test1():
    totals = []
    env = gym.make("CartPole-v0")

    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        for step in range(1000): # 1000 steps max
            action = basic_policy(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        totals.append(episode_rewards)

    print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))


# Neural network policy: REINFORCE policy gradient method
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


def drl_test2():
    # Construct the network structure
    n_inputs = 4
    n_hidden1 = 4
    n_hidden2 = 2
    n_outputs = 1
    initializer = tf.contrib.layers.variance_scaling_initializer() # He's initialization

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden1 = fully_connected(X, n_hidden1, activation_fn=tf.nn.relu, weights_initializer=initializer)
    hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=tf.nn.relu, weights_initializer=initializer)
    logits = fully_connected(hidden2, n_outputs, activation_fn=None, weights_initializer=initializer)
    output = tf.nn.sigmoid(logits)

    p_left_and_right = tf.concat(axis=1, values=[output, 1-output])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    init = tf.global_variables_initializer()

    y = 1 - tf.to_float(action) # label of the sampled action

    learning_rate = 0.01

    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels= y, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_var = optimizer.compute_gradients(xentropy)

    gradients = [grad for grad, variable in grads_and_var]

    # Hold the gradients of each evaluation
    gradient_placehoders = []
    grads_and_var_feed = []
    for grad, variable in grads_and_var:
        gradient_placehoder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placehoders.append(gradient_placehoder)
        grads_and_var_feed.append((gradient_placehoder, variable))

    training_op = optimizer.apply_gradients(grads_and_var_feed)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Training
    n_iterations = 500
    n_max_steps = 200
    n_games_per_update = 10
    save_iterations = 50
    discount_rate = 0.95

    env = gym.make("CartPole-v0")
    with tf.Session() as sess:
        sess.run(init)
        for iteration in range(n_iterations):
            all_rewards = []
            all_gradients = []
            mean_game_reward = []
            for game in range(n_games_per_update):
                current_rewards = []
                current_gradients = []

                obs = env.reset()

                for step in range(n_max_steps):
                    action_val, gradients_val = sess.run(
                        [action, gradients],
                        feed_dict= {X: obs.reshape(1, n_inputs)})
                    obs, reward, done, info = env.step(action_val[0][0])
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    if done:
                        break

                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

                mean_game_reward.append(np.sum(current_rewards))

            print("Iteration: " + str(iteration))
            print(np.mean(mean_game_reward), np.std(mean_game_reward), np.min(mean_game_reward), np.max(mean_game_reward))

            # Update the gradients
            all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
            feed_dict = {}
            for var_index, grad_placehoder in enumerate(gradient_placehoders):
                # Multiply the gradients by the action scores and compute the mean
                mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                          for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
                feed_dict[grad_placehoder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)
            if iteration % save_iterations == 0:
                saver.save(sess, "model/policy_net-"+str(iteration))

    env.close()


# Deep Q-learning with Atari Ms. Pac-Man game
def drl_test3():
    from tensorflow.contrib.layers import convolution2d, fully_connected
    from collections import deque
    import os

    # Preprocessing the image
    mspacman_color = np.array([210, 164, 74]).mean()

    def preprocess_observation(obs):
        img = obs[1:176:2, ::2]
        img = img.mean(axis=2)
        img[img == mspacman_color] = 0  # improve the contrast
        img = (img - 128) / 128 - 1  # from -1 to 1
        return img.reshape(88, 80, 1)


    # Set-up the Atari environment
    env = gym.make("MsPacman-v0")

    # Parameters
    input_height = 88
    input_width = 80
    input_channels = 1
    conv_n_maps = [32, 64, 64]
    conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
    conv_strides = [4, 2, 1]
    conv_paddings = ["SAME"] * 3
    conv_activation = [tf.nn.relu] * 3
    n_hidden_in = 64 * 11 * 10
    n_hidden = 512
    hidden_activation = tf.nn.relu
    n_outputs = env.action_space.n
    initializer = tf.contrib.layers.variance_scaling_initializer()

    learning_rate = 0.01

    # Used to create two Q networks, one is to provide temporary policy, the other perform optimization (double Q-learning)
    def q_network(X_state, scope):
        prev_layer = X_state
        conv_layers = []
        # Build all layers using a loop
        with tf.variable_scope(scope) as scope:
            for n_maps, kernel_size, stride, padding, activation in zip(conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
                prev_layer = convolution2d(prev_layer, num_outputs=n_maps, kernel_size=kernel_size, stride=stride,
                                           padding=padding, activation_fn=activation, weights_initializer=initializer)
                conv_layers.append(prev_layer)
            last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
            hidden = fully_connected(last_conv_layer_flat, num_outputs=n_hidden, activation_fn=hidden_activation, weights_initializer=initializer)
            outputs = fully_connected(hidden, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        # Name to variable map, strip out the scope name
        trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}

        return outputs, trainable_vars_by_name

    # Construct the model
    X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
    # The one who provide temporary policy (online)
    online_q_values, online_vars = q_network(X_state, "q_network/online")
    # The one perform optimization (target)
    target_q_values, target_vars = q_network(X_state, "q_network/target")

    # Periodically copy optimized values (target) to the temporary Q-network (online)
    copy_ops = [target_var.assign(online_vars[var_name]) for var_name, target_var in target_vars.items()]
    copy_online_to_target = tf.group(*copy_ops) # group all copy operation to a single operation

    learning_rate = 0.001
    momentum = 0.95

    with tf.variable_scope("train"):
        X_action = tf.placeholder(tf.int32, shape=[None])
        y = tf.placeholder(tf.float32, shape=[None, 1])
        q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                                axis=1, keep_dims=True)
        # Tweak to make the error being a squared error at [0,1] plus the linear part if greater than 1.0
        error = tf.abs(y - q_value)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error) # times 2 is to make the gradient continuous
        loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        training_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    # Set up the replay buffer
    replay_memory_size = 10000
    replay_memory = deque([], maxlen=replay_memory_size)

    def sample_memories(batch_size):
        indices = np.random.permutation(len(replay_memory))[:batch_size]
        cols = [[], [], [], [], []] # state, action, reward, next_state, continue
        for idx in indices:
            memory = replay_memory[idx]
            # Interesting implementation
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols] # Convert to numpy array
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

    # Set-up epsilon-greedy policy: gradually reduce from 1 to 0.05 in 50,000 training steps
    eps_min = 0.05
    eps_max = 1.0
    eps_decay_step = 100000

    def epsilon_greedy(q_values, step):
        epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_step)
        if np.random.rand() < epsilon:
            return np.random.randint(n_outputs)
        else:
            return np.argmax(q_values) # optimal action

    # Training section
    # Parameters
    n_steps = 1000000        # N training steps
    training_start = 1000    # start training after 1000 game iterations
    training_interval = 4   # run training step every 4 game iterations
    save_steps = 1000        # save every some steps
    copy_steps = 1000         # copy target to online network period
    discount_rate = 0.95
    skip_start = 90         # skip the start of every game
    batch_size = 50
    iteration = 0           # game iteration
    checkpoint_path = "model/DQN"
    done = True             # env needs to be reset

    # Parameters to track the game performance
    loss_val = np.infty
    game_length = 0
    total_max_q = 0
    mean_max_q = 0.0

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_path + ".index"):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()
            copy_online_to_target.run()
        while True:
            step = global_step.eval()
            if step >= n_steps:
                break
            iteration += 1
            print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
                iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")

            if done: # if game over, start again
                obs = env.reset()
                # skip the start of each game
                for skip in range(skip_start):
                    obs, reward, done, info = env.step(0)
                state = preprocess_observation(obs)

            # Online step
            q_value = online_q_values.eval(feed_dict={X_state: [state]})
            action = epsilon_greedy(q_value, step)

            # Play
            obs, reward, done, info = env.step(action)
            next_state = preprocess_observation(obs)

            # Store memory
            replay_memory.append((state, action, reward, next_state, 1.0 - done))
            state = next_state

            # Compute statistics for tracking progress
            total_max_q += q_value.max()
            game_length += 1
            if done:
                mean_max_q = total_max_q / game_length
                total_max_q = 0.0
                game_length = 0

            if iteration < training_start or iteration % training_interval != 0:
                continue

            # Sample memories and use the target DQN to produce the target Q-Value
            X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
            next_q_values = target_q_values.eval(feed_dict={X_state: X_next_state_val})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues * discount_rate * max_next_q_values

            # Train the online DQN
            _, loss_val = sess.run([training_op, loss], feed_dict={
                X_state: X_state_val, X_action: X_action_val, y: y_val})

            # Regularly copy the online DQN to the target DQN
            if step % copy_steps == 0:
                copy_online_to_target.run()

            # And save regularly
            if step % save_steps == 0:
                saver.save(sess, checkpoint_path)

def main():
    drl_test3()

if __name__ == "__main__":
    main()