import tensorflow as tf


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.02)
  #initial = tf.zeros(shape, dtype=tf.float32)
  return tf.Variable(initial)

def bias_variable(shape):
  #initial = tf.constant(0.00000000001, shape=shape)
  initial = tf.zeros(shape, dtype=tf.float32)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# Model:

learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

 # 51 height, 80 width, 2 channel depth. Grayscale and Motion channels
x_state = tf.placeholder(tf.float32, [None, 102, 160, 2])

# 8 possible actions were defined for this agent on 'DuskDrive' environmnet.
Q_values_new = tf.placeholder(tf.float32, shape=[None, 8])

    # 1st Convolutional layer
# Weights(shared within a filter) for first layer.
# 5x5 filter size, 2 channels, 32 different filters (size of the depth column).
# All of them initialized with positive values.
W_conv1 = weight_variable([8, 8, 2, 32])
b_conv1 = bias_variable([32])
# Only 32 because we have 32 filters. Each bias is for one filter.

h_conv1 = tf.nn.relu(conv2d(x_state, W_conv1) + b_conv1) # Size 102x160x32
h_pool1 = max_pool_2x2(h_conv1) # Finished 1st Conv. layer after ReLU and Max pooling. # Size 51x80x32

    # 2st Convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # Size 51x80x64
h_pool2 = max_pool_2x2(h_conv2) # Finished 2nd Conv. layer. Should result in 26x40x64 volume.

    # 3rd Convolutional layer
W_conv3 = weight_variable([4, 4, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) # Size 26x40x64
h_pool3 = max_pool_2x2(h_conv3) # Finished 2nd Conv. layer. Should result in 13x20x128 volume.


    # 1st Densly connected layer
W_fc1 = weight_variable([13 * 20 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 13*20*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout
# Prevents over-fitting. Use less than 1.0 only when training. When evaluating leave 1.0.
keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 2nd Densly connected layer
W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # 3rd Densly connected layer
W_fc3 = weight_variable([1024, 1024])
b_fc3 = bias_variable([1024])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)


    # 4th Densly connected layer
W_fc4 = weight_variable([1024, 1024])
b_fc4 = bias_variable([1024])

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)



    # Readout layer
W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])

Q_values_est = tf.matmul(h_fc4_drop, W_fc2) + b_fc2 # Output layer. A vector of 8 floats.
#tf.summary.scalar('Q_values', Q_values_est)

squared_error = tf.square(Q_values_est - Q_values_new)
sum_squared_error = tf.reduce_sum(squared_error, axis=1)
loss = tf.reduce_mean(sum_squared_error)
#tf.summary.scalar('Loss', loss)

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)