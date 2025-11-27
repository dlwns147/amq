import copy
import torch
import torch.nn as nn
from amq.utils import get_correlation
import gc
from torch.nn import DataParallel
# import numpy as np
# import math
# from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    # N-layer MLP
    def __init__(self, n_feature, n_layers=2, n_hidden=300, n_output=1, drop=0.2, n_select=2):
        super(Net, self).__init__()

        # self.embedding = nn.Parameter(torch.tensor(n_feature, n_select, n_hidden))
        # self.stem = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU())
        self.stem = nn.Sequential(nn.Linear(n_feature, n_hidden), nn.ReLU())

        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)

        self.regressor = nn.Linear(n_hidden, n_output)  # output layer
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.stem(x)
        x = self.hidden(x)
        x = self.drop(x)
        x = self.regressor(x)  # linear output
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            n = m.in_features
            # y = 1.0 / np.sqrt(n)
            # m.weight.data.uniform_(-y, y)
            # m.bias.data.fill_(0)
            nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias.data)


class MLP:
    """ Multi Layer Perceptron """
    def __init__(self, device='cpu', **kwargs):
        self.model = Net(**kwargs)
        self.model.to(device)
        self.name = 'mlp'
        self.device = device

    def fit(self, **kwargs):
        self.model = train(self.model, **kwargs)

    def predict(self, test_data):
        return predict(self.model, test_data, device=self.device)


def train(net, x, y, trn_split=0.8, pretrained=None, device='cpu', batch_size=128,
          lr=8e-4, epochs=2000, verbose=False):

    n_samples = x.shape[0]
    target = torch.zeros(n_samples, 1)
    perm = torch.randperm(target.size(0))
    trn_idx = perm[:int(n_samples * trn_split)]
    vld_idx = perm[int(n_samples * trn_split):]

    inputs = torch.from_numpy(x).float()
    target[:, 0] = torch.from_numpy(y).float()

    # back-propagation training of a NN
    if pretrained is not None:
        print("Constructing MLP surrogate model with pre-trained weights")
        init = torch.load(pretrained, map_location='cpu')
        net.load_state_dict(init)
        best_net = copy.deepcopy(net)
    else:
        # print("Constructing MLP surrogate model with "
        #       "sample size = {}, epochs = {}".format(x.shape[0], epochs))

        # initialize the weights
        # net.apply(Net.init_weights)
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        # n_iter = math.ceil(len(trn_idx) / batch_size)
        # optimizer = torch.optim.AdamW(net.parameters(), lr=lr/n_iter)
        criterion = nn.SmoothL1Loss()
        # criterion = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=0)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=1e-8)

        best_loss = 1e33
        # trn_inputs, trn_labels = inputs[trn_idx], target[trn_idx]
        # val_inputs, val_labels = inputs[vld_idx], target[vld_idx]
        for epoch in range(epochs):
            # loss_trn = train_one_epoch(net, trn_inputs, trn_labels, criterion, optimizer, device, batch_size)
            # loss_trn = train_one_epoch(net, trn_inputs, trn_labels, criterion, optimizer, device)
            # loss_vld = infer(net, val_inputs, val_labels, criterion, device)

            
            trn_inputs = inputs[trn_idx]
            trn_labels = target[trn_idx]
            loss_trn = train_one_epoch(net, trn_inputs, trn_labels, criterion, optimizer, device)
            loss_vld = infer(net, inputs[vld_idx], target[vld_idx], criterion, device)
            scheduler.step()
            

            # if epoch % 500 == 0 and verbose:
            #     print("Epoch {:4d}: trn loss = {:.4E}, vld loss = {:.4E}".format(epoch, loss_trn, loss_vld))

            if loss_vld < best_loss:
                best_loss = loss_vld
                best_net = copy.deepcopy(net)

    # validate(best_net, inputs, target, device=device)
    
    gc.collect()
    torch.cuda.empty_cache()

    # return best_net.to('cpu')
    return best_net


def train_one_epoch(net, data, target, criterion, optimizer, device):
    net.train()
    optimizer.zero_grad()

    data, target = data.to(device), target.to(device)
    pred = net(data)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    return loss.item()


def infer(net, data, target, criterion, device):
    net.eval()

    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        pred = net(data)
        loss = criterion(pred, target)

    return loss.item()


def validate(net, data, target, device):
    net.eval()

    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        pred = net(data)
        pred, target = pred.cpu().detach().numpy(), target.cpu().detach().numpy()

        rmse, rho, tau = get_correlation(pred, target)

    # print("Validation RMSE = {:.4f}, Spearman's Rho = {:.4f}, Kendall’s Tau = {:.4f}".format(rmse, rho, tau))
    return rmse, rho, tau, pred, target


def predict(net, query, device):

    if query.ndim < 2:
        data = torch.zeros(1, query.shape[0])
        data[0, :] = torch.from_numpy(query).float()
    else:
        data = torch.from_numpy(query).float()

    net = net.to(device)
    net.eval()
    with torch.no_grad():
        data = data.to(device)
        pred = net(data)

    return pred.cpu().detach().numpy()


# def train_one_epoch(net, data, target, criterion, optimizer, device, batch_size):
#     net.train()

#     dataset = TensorDataset(data, target)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     for data, target in dataloader:
        # data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         pred = net(data)
#         loss = criterion(pred, target)
#         loss.backward()
#         optimizer.step()

#     return loss.item()