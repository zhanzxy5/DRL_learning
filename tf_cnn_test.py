import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def cnn_test1():
    from sklearn.datasets import load_sample_images

    # Load sample images
    dataset = np.array(load_sample_images().images, dtype=np.float32)
    batch_size, height, width, channels = dataset.shape

    # create 2 filters
    filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
    filters_test[:, 3, :, 0] = 1
    filters_test[3, :, :, 1] = 1

    # Create the graph with input X plus a convolutional layer applying the 2 filters
    X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    convolution = tf.nn.conv2d(X, filters_test, strides=[1, 2, 2, 1], padding="SAME")
    relu = tf.nn.relu(convolution)
    max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.Session() as sess:
        output = sess.run(max_pool, feed_dict={X: dataset})

    plt.imshow(output[0, :, :, 1])
    plt.show()


def cnn_test2():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/")

    height = 28
    width = 28
    channels = 1
    n_inputs = height * width

    conv1_fmaps = 32
    conv1_ksize = 3
    conv1_stride = 1
    conv1_pad = "SAME"

    conv2_fmaps = 64
    conv2_ksize = 3
    conv2_stride = 2
    conv2_pad = "SAME"

    pool3_fmaps = conv2_fmaps

    n_fc1 = 64
    n_outputs = 10

    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
        X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
        y = tf.placeholder(tf.int32, shape=[None], name="y")

    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                             strides=conv2_stride, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv2")

    with tf.name_scope("pool3"):
        pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])

    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

    with tf.name_scope("output"):
        logits = tf.layers.dense(fc1, n_outputs, name="output")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    n_epochs = 10
    batch_size = 100

    # Sample 1000 images for testing
    X_test = mnist.test.images
    indices = np.random.permutation(len(X_test))[:1000]
    X_test = X_test[indices]
    y_test = mnist.test.labels[indices]

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

            save_path = saver.save(sess, "model/my_mnist_model")


def main():
    cnn_test2()

if __name__ == "__main__":
    main()