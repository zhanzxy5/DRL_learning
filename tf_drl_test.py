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





def main():
    drl_test2()

if __name__ == "__main__":
    main()