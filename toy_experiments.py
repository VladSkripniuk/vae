import os
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
from tqdm import tqdm

import numpy as np

from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F


from models	import VAE_2_n_2, VAE_loss_gaussian_x, VAE_loss_binary_x
from datasets import Blobs, Bananas
from logger import Logger

parser = argparse.ArgumentParser(description='VAE on toy data')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dir', default=None,
                    help='')
args = parser.parse_args()

with Logger(args.dir, __file__) as logger:

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(42)
    torch.manual_seed(42)


    x_train = Blobs(n_samples=1000)
    x_test = Blobs(n_samples=100)

    logger.put('x_train', np.array(x_train))
    logger.put('x_test', np.array(x_test))

    train_loader = DataLoader(x_train, batch_size=50, shuffle=True)
    test_loader = DataLoader(x_train, batch_size=50, shuffle=True)

    model = VAE_2_n_2().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = VAE_loss_gaussian_x(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

    def test(epoch):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += VAE_loss_gaussian_x(recon_batch, data, mu, logvar).item()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


    for epoch in tqdm(range(1, 50 + 1), file=sys.__stdout__):
        train(epoch)
        test(epoch)

        with torch.no_grad():
            sample = torch.randn(1000, 2).to(device)
            sample = model.decode(sample).cpu()
            sample = sample.numpy()

        logger.put("samples_epoch_{}".format(epoch), sample)
