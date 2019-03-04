import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
args = parser.parse_args()


dat = np.loadtxt(args.file)

test_accs = {}
train_accs = {}

for i in range(dat.shape[0]):
	if dat[i, 0] not in test_accs:
		test_accs[dat[i,0]] = []
		train_accs[dat[i,0]] = []
	test_accs[dat[i,0]].append(dat[i,2])
	train_accs[dat[i,0]].append(dat[i,1])

test_means = { k : np.mean(test_accs[k]) for k in test_accs }
train_means = { k : np.mean(train_accs[k]) for k in train_accs }

test_stds = { k : np.std(test_accs[k]) for k in test_accs }
train_stds = { k : np.std(train_accs[k]) for k in train_accs }

for k in test_means:
	print(k, train_means[k], train_stds[k], test_means[k], test_stds[k])