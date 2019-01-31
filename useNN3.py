import numpy as np
import matplotlib.pyplot as plt
from MNIST_NeuralNetwork import MNIST_NeuralNetwork
from load_mnist import load_mnist

train_x, train_y = load_mnist('mnist', kind='train')

test_x, test_y = load_mnist('mnist', kind='t10k')

nn = MNIST_NeuralNetwork(l2 = 0.1,
                         l1 = 0.0,
                         epochs = 10,
                         eta = 0.001,
                         alpha = 0.001,
                         decrease_const = 0.00001,
                         shuffle = True,
                         minibatches = 50,
                         random_state = 1)

nn.fit(train_x, train_y, print_progress = True)


train_y_pred = nn.predict(train_x)
acc = np.sum(train_y == train_y_pred, axis = 0) / train_x.shape[0]
print('Training accuracy: {:.2f}%'.format((acc * 100)))


test_y_pred = nn.predict(test_x)
acc = np.sum(test_y == test_y_pred, axis = 0) / test_x.shape[0]
print('Test accuracy: {:.2f}%'.format((acc * 100)))
