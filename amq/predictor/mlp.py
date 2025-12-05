import gc
import copy

import torch
import torch.nn as nn

from utils.func import get_correlation

class Net(nn.Module):
    # N-layer MLP
    def __init__(self, n_feature, n_layers=2, n_hidden=300, n_output=1, drop=0.2, n_select=2):
        super(Net, self).__init__()

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
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=0)

        best_loss = 1e33
        for epoch in range(epochs):
            
            trn_inputs = inputs[trn_idx]
            trn_labels = target[trn_idx]
            loss_trn = train_one_epoch(net, trn_inputs, trn_labels, criterion, optimizer, device)
            loss_vld = infer(net, inputs[vld_idx], target[vld_idx], criterion, device)
            scheduler.step()
            
            if loss_vld < best_loss:
                best_loss = loss_vld
                best_net = copy.deepcopy(net)

    gc.collect()
    torch.cuda.empty_cache()

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
