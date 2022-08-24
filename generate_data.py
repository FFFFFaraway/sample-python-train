import numpy as np

N = 1000
ratio = 0.8

a = np.random.uniform(-10, 10, N)
b = np.random.uniform(-10, 10, N)
noise = np.random.uniform(0, 1, N)
c = a + b + a * b + noise
data = np.stack([a, b, c], axis=-1)

pos = int(N * ratio)
train_set = data[:pos]
test_set = data[pos:]

# local save
np.savetxt('train.csv', train_set, delimiter=',')
np.savetxt('test.csv', test_set, delimiter=',')
