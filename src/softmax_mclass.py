import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import label_binarize
from IPython.display import display

df = pd.read_csv('/tmp/winequality-red.csv', sep=';')
cols = [c.replace(' ', '_').lower() for c in df.columns]
df.columns = cols
df['y'] = df['quality'].apply(lambda x: 2 if x > 5  else 1 if x > 4 else 0)
X = df[[c for c in df.columns if c != 'y']]
y = label_binarize(df['y'], classes = sorted(df['y'].unique()))
num_labels = y.shape[1]
# Train-test initialization
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))
X_train = X.values[train_index]
X_test = X.values[test_index]
y_train = y[train_index]
y_test = y[test_index]
print('y_train: {}, y_test: {}'.format(y_train.shape, y_test.shape))
X_train_norm = (X_train - X_train.mean())/X_train.std()
X_test_norm = (X_test - X_train.mean())/X_train.std()
# Model definition
W = tf.Variable(tf.random_normal(shape=[X_train_norm.shape[1], num_labels]))
b = tf.Variable(tf.random_normal(shape=[1, num_labels]))
data = tf.placeholder(dtype=tf.float32, shape=[None, X_train_norm.shape[1]])
target = tf.placeholder(dtype=tf.float32, shape=[None, num_labels])
mod = tf.matmul(data, W) + b
# Loss function (logistic with L2 regularization)
learning_rate = 0.03
batch_size = 128
num_epochs = 50
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target)) + 0.01*tf.nn.l2_loss(W)
obj = tf.train.AdamOptimizer(learning_rate).minimize(loss)
prediction = tf.nn.softmax(mod)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1)), dtype=tf.float32))
# Train model
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        batch_index = np.random.choice(len(X_train_norm), size=batch_size)
        X_train_norm_batch = X_train_norm[batch_index]
        y_train_batch = y_train[batch_index]
        sess.run(obj, feed_dict={data: X_train_norm_batch, target: y_train_batch})
        temp_loss = sess.run(loss, feed_dict={data: X_train_norm_batch, target: y_train_batch})
        preds = sess.run(tf.argmax(prediction, 1), feed_dict={data: X_test_norm})
        actuals = sess.run(tf.argmax(y_test, 1))
        temp_train_acc = sess.run(accuracy, feed_dict={data: X_train_norm, target: y_train})
        temp_test_acc = sess.run(accuracy, feed_dict={data: X_test_norm, target: y_test})
        if(epoch%5 == 0):
            print ('epoch: {}, loss: {}, train_acc: {}, test_acc: {}'.format(epoch, temp_loss, temp_train_acc, temp_test_acc))

# Display predictions
print('Predictions Summary')
pdf = pd.DataFrame([(a, p) for a, p in zip(actuals, preds)], columns = ['actual', 'predicted'])
display(pdf)