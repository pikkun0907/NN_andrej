import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.softmax(h, dim=1)
        return h


@torch.no_grad()
def calc_accuracy(data_loader):
    accuracy = 0

    for batch in data_loader:
        x, t = batch
        x = x.to(device)
        t = t.to(device)
        y_pred = net(x)

        y_pred_label = torch.argmax(y_pred, dim=1)
        accuracy += (y_pred_label == t).sum().float()/len(t)

    accuracy /= len(data_loader)
    return accuracy

    # dataset preparation
iris = load_iris()
x = iris['data']
t = iris['target']
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

dataset = torch.utils.data.TensorDataset(x, t)

# train : val : test = 60 : 20 : 20
n_train = int(len(dataset)*0.6)
n_val = int(len(dataset)*0.2)
n_test = len(dataset) - n_train - n_val

train, val, test = torch.utils.data.random_split(
    dataset, [n_train, n_val, n_test])

# batch
batch_size = 10
train_loader = torch.utils.data.DataLoader(
    train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)


max_epoch = 1000
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net = Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


for epoch in range(max_epoch):
    for batch in train_loader:
        x, t = batch
        x = x.to(device)
        t = t.to(device)

        y_pred = net(x)

        loss = F.cross_entropy(y_pred, t)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    print(f'loss: {loss}')


print(calc_accuracy(val_loader))
