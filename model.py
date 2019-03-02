import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, n):
        super(Model, self).__init__()


        channels = 10

        self.w1 = nn.Parameter(torch.Tensor(n, n))
        self.b1 = nn.Parameter(torch.Tensor(n))
        self.bn1 = nn.BatchNorm1d(n)

        self.w2 = nn.Parameter(torch.Tensor(n, n))
        self.b2 = nn.Parameter(torch.Tensor(n))
        self.bn2 = nn.BatchNorm1d(n)

        self.w3 = nn.Parameter(torch.Tensor(n, n))
        self.b3 = nn.Parameter(torch.Tensor(n))
        self.bn3 = nn.BatchNorm1d(n)

        self.w4 = nn.Parameter(torch.Tensor(n, n))
        self.b4  = nn.Parameter(torch.Tensor(n))
        self.bn4 = nn.BatchNorm1d(n)


        self.w5 = nn.Parameter(torch.Tensor(n, n))
        self.b5 = nn.Parameter(torch.Tensor(n))
        self.bn5 = nn.BatchNorm1d(n)

        self.w6 = nn.Parameter(torch.Tensor(n, n))
        self.b6 = nn.Parameter(torch.Tensor(n))
        self.bn6 = nn.BatchNorm1d(n)

        self.w7 = nn.Parameter(torch.Tensor(n, n))
        self.b7 = nn.Parameter(torch.Tensor(n))
        self.bn7 = nn.BatchNorm1d(n)

        self.w8 = nn.Parameter(torch.Tensor(n, n))
        self.b8  = nn.Parameter(torch.Tensor(n))
        self.bn8 = nn.BatchNorm1d(n)



        self.lin5 = nn.Linear(n, channels)
        self.lin6 = nn.Linear(channels, 1)

        self.reset_params()


    def reset_params(self):

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)
        nn.init.xavier_uniform_(self.w4)

        self.b1.data.fill_(0.01)
        self.b2.data.fill_(0.01)
        self.b3.data.fill_(0.01)
        self.b4.data.fill_(0.01)



    def forward(self, x):
        start = torch.ones((x.size(0), x.size(1)))

        o = self.bn1(F.relu((start.unsqueeze(2) * x * self.w1).sum(1) + self.b1))
        o = self.bn2(F.relu((o.unsqueeze(2) * x * self.w2).sum(1) + self.b2))
        o = self.bn3(F.relu((o.unsqueeze(2) * x * self.w3).sum(1) + self.b3))
        o = self.bn4(F.relu((o.unsqueeze(2) * x * self.w4).sum(1) + self.b4))

        o = self.bn1(F.relu((start.unsqueeze(2) * x * self.w5).sum(1) + self.b5))
        o = self.bn2(F.relu((o.unsqueeze(2) * x * self.w6).sum(1) + self.b6))
        o = self.bn3(F.relu((o.unsqueeze(2) * x * self.w7).sum(1) + self.b7))
        o = self.bn4(F.relu((o.unsqueeze(2) * x * self.w8).sum(1) + self.b8))
        o = F.relu(self.lin5(o))
        o = self.lin6(o)

        return torch.squeeze(o)