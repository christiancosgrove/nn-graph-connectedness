import os
import numpy as np
import argparse
from load import load_splits
from torch.optim import Adam
import torch
from torch import nn



parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='./data')
args = parser.parse_args()



x, y, x_t, y_t = load_splits(os.path.join(args.data), 30000)

print('baseline test acc ', y_t.mean())

from model import Model

model = Model(x.shape[1])

o = Adam(model.parameters(), lr=1e-3)

mb_size = 16




def eval_model():

    pred = torch.sigmoid(model(torch.Tensor(x_t))) > 0.5

    pred = pred.cpu().detach().numpy()

    err = np.abs(pred - y_t).mean()
    return err




i = 0
it = 0
while True:

    batch_x, batch_y = x[i : i + mb_size], y[i : i + mb_size]

    batch_x = torch.Tensor(batch_x)
    batch_y = torch.Tensor(batch_y)

    out_x = model(batch_x)
    loss = nn.BCEWithLogitsLoss()(out_x, batch_y)

    o.zero_grad()

    loss.backward()

    o.step()

    if it % 1000 == 0:
        print('loss ', loss, ' ', it)
        print(eval_model())

    i += mb_size
    if i >= x.shape[0]:
        i = 0
    it += 1