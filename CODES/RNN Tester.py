import numpy as np
import matplotlib.pyplot as plt
from MyRNN import *


X_t = np.arange(-10, 10, 0.1)
X_t = X_t.reshape(len(X_t), 1)
Y_t = np.sin(X_t) + 0.1 * np.random.randn(len(X_t), 1)

plt.plot(X_t, Y_t)
plt.show()

'''

rnn = RunRNN(X_t, Y_t, Tanh())

X_new = np.arange(0, 20, 0.3)
X_new = X_new.reshape(len(X_new), 1)

Y_hat = ApplyRNN(X_new, rnn)

plt.plot(X_t, Y_t)
plt.plot(X_new, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()
'''


dt = 120
rnn = RunRNN(Y_t, Y_t, Tanh(), n_epoch=1000, n_neurons=100, decay=0.1, dt=dt)

Y_hat = ApplyRNN(Y_t, rnn)

X_t = np.arange(len(Y_t))

plt.plot(X_t, Y_t)
plt.plot(X_t + dt, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()
