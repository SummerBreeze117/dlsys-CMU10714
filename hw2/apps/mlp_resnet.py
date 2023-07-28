import sys

sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
from tqdm import tqdm

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    main = nn.Sequential(norm(dim),
                         nn.ReLU(),
                         nn.Linear(dim, hidden_dim),
                         norm(hidden_dim),
                         nn.ReLU(),
                         nn.Linear(hidden_dim, dim))
    return nn.Residual(main)
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Linear(dim, hidden_dim),
                         nn.ReLU(),
                         *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                         nn.Linear(hidden_dim, num_classes))
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # tqdm
    N = len(dataloader.dataset)

    loss_func = nn.SoftmaxLoss()
    total_loss, total_err = [], 0.0
    # training
    if opt is not None:
        pbar = tqdm(total=N // dataloader.batch_size)
        for X, y in dataloader:
            logits = model(X.reshape((X.shape[0], -1)))

            loss = loss_func.forward(logits, y)
            total_loss.append(loss.numpy())
            total_err += np.sum(logits.numpy().argmax(axis=1) != y.numpy())

            opt.reset_grad()
            loss.backward()
            opt.step()
            loss.detach()

            # update tqdm
            pbar.update(1)
        pbar.close()
    # eval
    else:
        for X, y in dataloader:
            logits = model(X.reshape((X.shape[0], -1)))

            loss = loss_func.forward(logits, y)
            total_loss.append(loss.numpy())
            total_err += np.sum(logits.numpy().argmax(axis=1) != y.numpy())

    return total_err / N, np.mean(total_loss)
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz",
                                      f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = ndl.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                                     f"{data_dir}/t10k-labels-idx1-ubyte.gz")

    resnet = MLPResNet(28 * 28, hidden_dim=hidden_dim, num_classes=10)

    # train
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_dataloader = ndl.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for _ in range(1, epochs + 1):
        train_err, train_loss = epoch(train_dataloader, resnet, opt)
        print(f"acc: {1.0 - train_err:.4f}, Loss: {train_loss:.4f}")

    # eval
    test_dataloader = ndl.data.DataLoader(test_set, batch_size=batch_size)
    test_err, test_loss = epoch(test_dataloader, resnet)
    print(f"acc: {1.0 - train_err:.4f}, Loss: {test_loss:.4f}")
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION
