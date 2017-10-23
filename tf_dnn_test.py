import tensorflow as tf
import numpy as np

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def transform_housing():
    fpath = 'D:/Courses/TensorFlow/data/CaliforniaHousing/cal_housing.data'
    with open(fpath, 'r') as f:
        cal_housing = np.loadtxt(f, delimiter=',')
    # Columns are not in the same order compared to the previous
    # URL resource on lib.stat.cmu.edu
    columns_index = [8, 7, 2, 3, 4, 5, 6, 1, 0]
    cal_housing = cal_housing[:, columns_index]

    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                     "Population", "AveOccup", "Latitude", "Longitude"]

    target, data = cal_housing[:, 0], cal_housing[:, 1:]

    # avg rooms = total rooms / households
    data[:, 2] /= data[:, 5]

    # avg bed rooms = total bed rooms / households
    data[:, 3] /= data[:, 5]

    # avg occupancy = population / households
    data[:, 5] = data[:, 4] / data[:, 5]

    # target in units of 100,000
    target = target / 100000.0

    return (target, data)

def test1():
    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x * x * y + y + 2

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        result = f.eval()

    print(result)

# Linear regression using normal equation
def test2():
    (housing_Y, housing_X) = transform_housing()
    m, n = housing_X.shape
    housing_data_plus_bias = np.c_[np.ones((m,1)), housing_X]

    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name= "X")
    y = tf.constant(housing_Y.reshape(-1,1), dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        theta_value = theta.eval()

    print(theta_value)

# Linear regression with stochastic gradient descent
def test3():
    from sklearn.preprocessing import StandardScaler
    n_epochs = 1000
    learning_rate = 0.01

    # Get data
    (housing_Y, housing_X) = transform_housing()
    m, n = housing_X.shape
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing_X]

    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing_X)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing_Y.reshape(-1,1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n+1,1],-1,1), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

    # manual gradients
    # gradients = 2 / m * tf.matmul(tf.transpose(X), error)
    # training_op = tf.assign(theta, theta - learning_rate * gradients)

    # auto diff
    # gradients = tf.gradients(mse, [theta])[0]
    # training_op = tf.assign(theta, theta - learning_rate * gradients)

    # Using optimizer
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch ", epoch, "\tMSE = ", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()

    print(best_theta)

# Linear regression using mini-batch stochastic gradient descent
def test4():
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime

    n_epochs = 1000
    learning_rate = 0.01
    batch_size = 100

    # Get data
    (housing_Y, housing_X) = transform_housing()
    m, n = housing_X.shape
    n_batches = int(np.ceil(m / batch_size))

    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing_X)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    X = tf.placeholder(tf.float32, shape=(batch_size, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="y")

    theta = tf.Variable(tf.random_uniform([n+1,1],-1,1), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")

    # Use name scope to group variables
    with tf.name_scope("loss") as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name="mse")

    # Using optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    # Save the model
    saver = tf.train.Saver()
    save_filename = "model/test4_model.ckpt"

    # Set up the log summary
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logDir = "model/logs"
    logdir = "{}/run-{}".format(root_logDir, now)

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)
        indices = np.random.randint(m, size=batch_size)
        X_batch = scaled_housing_data_plus_bias[indices]
        y_batch = housing_Y.reshape(-1,1)[indices]
        return X_batch, y_batch

    with tf.Session() as sess:
        sess.run(init)
        # Just to make this not a local variable
        X_batch, y_batch = fetch_batch(0, 0, batch_size)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                if epoch % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
            if epoch % 100 == 0:
                print("Epoch ", epoch, "\tMSE = ", mse.eval(feed_dict={X: X_batch, y: y_batch}))
                # save_file = saver.save(sess, save_filename)

        best_theta = theta.eval()
        save_file = saver.save(sess, 'model/test4_model_final')
        print(save_file)

    file_writer.close()

    print(best_theta)

def test4_restore():
    saver = tf.train.import_meta_graph('model/test4_model_final.meta')
    theta = tf.get_default_graph().get_tensor_by_name("theta:0")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        best_theta_restored = theta.eval()
        print(best_theta_restored)



# Simple deep neural network
def fetch_batch(epoch, batch_index, batch_size, n_batches, m, X_data, y_data):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = X_data[indices]
    y_batch = y_data[indices]
    return X_batch, y_batch

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

# Selu activation function
def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))

def test5():
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]

    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name = "y")

    with tf.name_scope("DNN"):
        hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation=tf.nn.elu)
        hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation=tf.nn.elu)
        logits = neuron_layer(hidden2, n_outputs, "outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y, logits = logits)
        loss = tf.reduce_mean(xentropy, name= "loss")

    learning_rate = 0.01


    with tf.name_scope("train"):
        # Gradient descent optimizer
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Momoentum optimizer
        # optimizer = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum=0.9)
        # Nesterov Accellerated gradient optimizer
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov= True)
        # RMSProp optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9, decay=0.9, epsilon=1e-10)
        # Adam optimization
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Execution
    n_epochs = 400
    batch_size = 50
    m, n = X_train.shape
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size, n_batches, m, X_train, y_train)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict= {X: X_test, y: y_test})

            print(epoch, "Train accuracy: ", acc_train, "\tTest accuracy: ", acc_test)
        save_path = saver.save(sess, 'model/DNN')


# Batch normalization
def test6():
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import StandardScaler
    from tensorflow.contrib.layers import batch_norm
    from tensorflow.contrib.layers import fully_connected

    scaler = StandardScaler()
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]

    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name = "y")

    is_training = tf.placeholder(tf.bool, shape=(), name= 'is_training')
    bn_params = {
        'is_training': is_training,
        'decay': 0.99,
        'updates_collections': None
    }

    # Low level API
    with tf.name_scope("DNN"):
        hidden1 = fully_connected(X, n_hidden1, scope="hidden1", normalizer_fn=batch_norm,
                                  normalizer_params=bn_params, activation_fn=tf.nn.elu)
        hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2", normalizer_fn=batch_norm,
                                  normalizer_params=bn_params, activation_fn=tf.nn.elu)
        logits = fully_connected(hidden2, n_outputs, scope="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y, logits = logits)
        loss = tf.reduce_mean(xentropy, name= "loss")

    learning_rate = 0.01

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Execution
    n_epochs = 400
    batch_size = 50
    m, n = X_train.shape
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size, n_batches, m, X_train, y_train)
                sess.run(training_op, feed_dict={is_training: True, X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={is_training: False, X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict= {is_training: False, X: X_test, y: y_test})

            print(epoch, "Train accuracy: ", acc_train, "\tTest accuracy: ", acc_test)
        save_path = saver.save(sess, 'model/DNN')

# Dropout
def test7():
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import StandardScaler
    from tensorflow.contrib.layers import dropout
    from tensorflow.contrib.layers import fully_connected

    scaler = StandardScaler()
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]

    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name = "y")

    is_training = tf.placeholder(tf.bool, shape=(), name= 'is_training')

    # Low level API
    keep_prob = 0.5
    with tf.name_scope("DNN"):
        X_drop = dropout(X, keep_prob=keep_prob, is_training=is_training)
        hidden1 = fully_connected(X_drop, n_hidden1, scope="hidden1", activation_fn=tf.nn.elu)
        hidden1_drop = dropout(hidden1, keep_prob=keep_prob, is_training=is_training)
        hidden2 = fully_connected(hidden1_drop, n_hidden2, scope="hidden2", activation_fn=tf.nn.elu)
        hidden2_drop = dropout(hidden2, keep_prob=keep_prob, is_training=is_training)
        logits = fully_connected(hidden2_drop, n_outputs, scope="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y, logits = logits)
        loss = tf.reduce_mean(xentropy, name= "loss")

    learning_rate = 0.01

    with tf.name_scope("train"):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Execution
    n_epochs = 400
    batch_size = 50
    m, n = X_train.shape
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size, n_batches, m, X_train, y_train)
                sess.run(training_op, feed_dict={is_training: True, X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={is_training: False, X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict= {is_training: False, X: X_test, y: y_test})

            print(epoch, "Train accuracy: ", acc_train, "\tTest accuracy: ", acc_test)
        save_path = saver.save(sess, 'model/DNN')

def main():
    test5()

if __name__ == "__main__":
    main()