import matplotlib.pyplot as plt
# Import Dependencies
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import tensorflow as tf

features_df = pd.read_csv('ds_midterm2018.csv', usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13])
features_df.head()

features_df.shape
features_df.describe()

labels_df = pd.read_csv('ds_midterm2018_label.csv', usecols=[1])

labels_df.head()
labels_df.shape
X_train = features_df
y_train = labels_df

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(data=scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

X_train = np.array(X_train)
y_train = np.array(y_train)

type(X_train), type(y_train)

lr = 0.01# 수정

# Number of epochs for which the model will run
epochs = 500 # 수정

X = tf.placeholder(tf.float32,[None,X_train.shape[1]])


# Labels
y = tf.placeholder(tf.float32,[None,1])

W = tf.Variable(tf.random_normal([13,13])) # 괄호안을 채워 넣으세요
print(W)
# Bias
b = tf.Variable(tf.random_normal([500, 13]))  # 괄호안을 채워 넣으세요

init = tf.global_variables_initializer()

y_hat = tf.add(tf.matmul(X, W), b)

# Loss Function
loss=  tf.reduce_max(y_hat)# LOSS 함수를 여기에

# Gradient Descent Optimizer to Minimize the Cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

loss_history = np.empty(shape=[1],dtype=float)

with tf.Session() as sess:
    # Initialize all Variables
    sess.run(init)
    err = sess.run(loss, feed_dict={X: X_train, y: y_train})
    print('Epoch: 0, Error: {0}'.format(err))

    for epoch in range(1, epochs):
        # Run the optimizer and the cost functions
        result = sess.run(optimizer, feed_dict={X: X_train, y: y_train})
        err = sess.run(loss, feed_dict={X: X_train, y: y_train})

        # Add the calculated loss to the array
        loss_history = np.append(loss_history, err)

        # Print the Loss/Error after every 100 epochs
        if epoch % 100 == 0:
            print('Epoch: {0}, Error: {1}'.format(epoch, err))

    print('Epoch: {0}, Error: {1}'.format(epoch + 1, err))

    # Values of Weight & Bias after Training
    new_W = sess.run(W)
    new_b = sess.run(b)

    # Predicted Labels
    y_pred = sess.run(y_hat, feed_dict={X: X_train})

    # Error: MSE or MAE
    # error = tf.metrics.mean_absolute_error  # 계산
    error = tf.metrics.mean_squared_error
print('Trained Weights: \n', new_W)

print('Trained Bias: \n', new_b)


plt.plot(range(len(loss_history)), loss_history)
plt.axis([0, epochs, 0, np.max(loss_history)])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs', fontsize=25)

print('Predicted Values: \n', y_pred)

print('Error [TF Session]: ', error)

plt.show()

