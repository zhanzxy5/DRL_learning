import tensorflow as tf
import numpy as np
import matplotlib
import sys, os
import matplotlib.pyplot as plt

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def save_fig(fig_id, tight_layout=True):
    PROJECT_ROOT_DIR = "."
    CHAPTER_ID = "autoencoders"
    path = os.path.join(PROJECT_ROOT_DIR, "data", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# A stacked MNIST autoencoder
def encoder_test1():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("data/mnist/")

    from functools import partial

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 150  # codings
    n_hidden3 = n_hidden1
    n_outputs = n_inputs

    learning_rate = 0.01
    l2_reg = 0.0001

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])

    he_init = tf.contrib.layers.variance_scaling_initializer()  # He initialization
    # Equivalent to:
    # he_init = lambda shape, dtype=tf.float32: tf.truncated_normal(shape, 0., stddev=np.sqrt(2/shape[0]))
    l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    my_dense_layer = partial(tf.layers.dense,
                             activation=tf.nn.elu,
                             kernel_initializer=he_init,
                             kernel_regularizer=l2_regularizer)

    hidden1 = my_dense_layer(X, n_hidden1)
    hidden2 = my_dense_layer(hidden1, n_hidden2)
    hidden3 = my_dense_layer(hidden2, n_hidden3)
    outputs = my_dense_layer(hidden3, n_outputs, activation=None)

    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([reconstruction_loss] + reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    n_epochs = 5
    batch_size = 150

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")  # not shown in the book
                sys.stdout.flush()  # not shown
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})  # not shown
            print("\r{}".format(epoch), "Train MSE:", loss_train)  # not shown


        n_test_digits = 10
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})
        fig = plt.figure(figsize=(8, 3 * n_test_digits))
        for digit_index in range(n_test_digits):
            plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
            plot_image(X_test[digit_index])
            plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
            plot_image(outputs_val[digit_index])

        save_fig("reconstruction_plot")

def main():
    encoder_test1()

if __name__ == "__main__":
    main()