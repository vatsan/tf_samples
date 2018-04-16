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

# Network Parameters
n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 32 # 2nd layer number of neurons
n_input = X_train.shape[1] # feature vector length
n_classes = y.shape[1] # 3 classes

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Model definition
data = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
target = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
mod = multilayer_perceptron(data)
learning_rate = 0.03
batch_size = 128
n_batches = X_train.shape[0]/batch_size
num_epochs = 50
# Loss function (logistic with L2 regularization)
regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out'])
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target)) + 0.01*regularizer
obj = tf.train.AdamOptimizer(learning_rate).minimize(loss)
prediction = tf.nn.softmax(mod)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1)), dtype=tf.float32))
# Train model
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        indices_shuffled = list(range(X_train_norm.shape[0]))
        np.random.shuffle(indices_shuffled)
        avg_loss = 0
        for batch in range(int(n_batches)):
            batch_index = indices_shuffled[batch: (batch+1)*batch_size]
            X_train_norm_batch = X_train_norm[batch_index]
            y_train_batch = y_train[batch_index]
            print('X_train_norm_batch : {}, y_train_batch: {}'.format(X_train_norm_batch.shape, y_train_batch.shape))
            sess.run(obj, feed_dict={data: X_train_norm_batch, target: y_train_batch})
            temp_loss = sess.run(loss, feed_dict={data: X_train_norm_batch, target: y_train_batch})
            avg_loss += temp_loss/n_batches
            preds = sess.run(tf.argmax(prediction, 1), feed_dict={data: X_test_norm})
            actuals = sess.run(tf.argmax(y_test, 1))
            temp_train_acc = sess.run(accuracy, feed_dict={data: X_train_norm, target: y_train})
            temp_test_acc = sess.run(accuracy, feed_dict={data: X_test_norm, target: y_test})
        print ('epoch: {}, avg_loss: {}, train_acc: {}, test_acc: {}'.format(epoch, avg_loss, temp_train_acc, temp_test_acc))

# Display predictions
print('Predictions Summary')
pdf = pd.DataFrame([(a, p) for a, p in zip(actuals, preds)], columns = ['actual', 'predicted'])
display(pdf)








