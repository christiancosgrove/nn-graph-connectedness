import numpy as np

def load_data(file):
    dat = np.genfromtxt(file, skip_header=7, dtype='str')

    g = dat.shape[0]

    e = dat.shape[1] - 1
    n = int((1 + np.sqrt(1 + 8 * e)) / 2)
    x = np.zeros((g, n, n))

    c = 0
    for i in range(n):
        for j in range(i+1, n):
            x[:, i, j] = dat[:, c]
            x[:, j, i] = dat[:, c]
            c+=1

    y = np.zeros(g)
    y[dat[:, e] == 'Y'] = 1
    return x,y

def load_splits(file, train_examples):

    x, y = load_data(file)

    print(x.shape[0])
    train_indices = np.random.choice(x.shape[0], size=train_examples, replace=False)

    test_indices = np.ones(x.shape[0], dtype=np.bool)
    test_indices[train_indices] = False

    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]