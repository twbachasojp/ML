import matplotlib.pyplot as plt
from load_mnist import load_mnist

train_x, train_y = load_mnist('mnist', kind='train')
print(train_x.shape)
print(train_y.shape)

test_x, test_y = load_mnist('mnist', kind='t10k')
print(test_x.shape)
print(test_y.shape)

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()

for i in range(25):
    img = train_x[train_y == 8][i].reshape(28, 28)

    ax[i].imshow(img, cmap='Greys', interpolation='nearest')


ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
