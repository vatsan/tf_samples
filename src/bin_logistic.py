import pandas as pd
import tensorflow as tf
import numpy as np
df = pd.read_csv('/tmp/winequality-red.csv', sep=';')
cols = [c.replace(' ', '_').lower() for c in df.columns]
df.columns = cols
df['y'] = df['quality']>5
df['y'] = df['y'].astype('int64')
X = df[[c for c in df.columns if c != 'y']]
y = df['y']

# Train-test initialization
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))
X_train = X.values[train_index]
X_test = X.values[test_index]
y_train = y.values[train_index]
y_test = y.values[test_index]
X_train_norm = (X_train - X_train.mean())/X_train.std()
X_test_norm = (X_test - X_train.mean())/X_train.std()

# Model definition
W = tf.Variable(tf.random_normal(shape=[X_train_norm.shape[1], 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
data = tf.placeholder(dtype=tf.float32, shape=[None, X_train_norm.shape[1]])
target = tf.placeholder(dtype=tf.float32, shape=[None, 1])
mod = tf.matmul(data, W) + b

# Loss function (logistic with L2 regularization)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target)) + 0.01*tf.nn.l2_loss(W)
learning_rate = 0.003
batch_size = 30
iter_num = 15000
obj = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
prediction = tf.round(tf.sigmoid(mod))
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target), dtype=tf.float32))

# Train model
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(iter_num):
        batch_index = np.random.choice(len(X_train_norm), size=batch_size)
        X_train_norm_batch = X_train_norm[batch_index]
        y_train_batch = np.matrix(y_train[batch_index]).T
        sess.run(obj, feed_dict={data: X_train_norm_batch, target: y_train_batch})
        temp_loss = sess.run(loss, feed_dict={data: X_train_norm_batch, target: y_train_batch})
        preds = sess.run(prediction, feed_dict={data: X_test_norm})
        temp_train_acc = sess.run(accuracy, feed_dict={data: X_train_norm, target: np.matrix(y_train).T})
        temp_test_acc = sess.run(accuracy, feed_dict={data: X_test_norm, target: np.matrix(y_test).T})
        if(epoch%100 ==0):
            print ('epoch: {}, loss: {}, train_acc: {}, test_acc: {}'.format(epoch, temp_loss, temp_train_acc, temp_test_acc))
            # print('actual: {}, predicted: {}'.format(y_test, preds))