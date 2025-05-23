from torch import device, cuda, no_grad, max
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64 * 7 * 7, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.output(x)

        return x


def mnist_uploader():
    _datasets = {
        "train_dataset": datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        ),
        "test_dataset": datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        ),
    }
    _data_loaders = {
        "train_loader": DataLoader(
            _datasets["train_dataset"], batch_size=batch_size, shuffle=True
        ),
        "test_loader": DataLoader(
            _datasets["test_dataset"], batch_size=batch_size, shuffle=False
        ),
    }
    return _datasets, _data_loaders


def train_model(model, train_loader, criterion, optimizer, epochs):
    losses = [1]
    for epoch in range(epochs):
        model.train()
        total_loss= 0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")
    return losses

# ============================================= #

if __name__ == "__main__":
   

    batch_size = 64
    epochs = 5
    learning_rate = 0.01
    momentum = 0.9

    device = device("cuda" if cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")
    # ============================================= #

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(), lr=learning_rate, momentum=momentum
    )
    ds, dl = mnist_uploader()
    # print(ds["train_dataset"][0])

    losses = train_model(model, dl["train_loader"], criterion, optimizer, epochs)

    model.eval()
    correct = 0
    total = 0
    with no_grad():
        for data, target in dl["test_loader"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

    x = range(epochs+1)

    fig, ax = plt.subplots()
    ax.plot(x, losses)

    fig.savefig('mnist/losses.png')
