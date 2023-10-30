import os
import argparse
import torch as th
import torchmetrics.functional as thm
from tqdm import tqdm
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay

from models import GCN, GAT, SAGE
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(args):
    device = th.device('cpu')
    trainset = UPFD(root='../data', name=args.dataset, feature=args.features, split='train')
    testset = UPFD(root='../data', name=args.dataset, feature=args.features, split='test')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    readout = scatter_mean if args.readout == 'mean' else scatter_add

    if args.model == 'gcn':
        model = GCN(trainset.num_features, args.nhids).to(device)
    elif args.model == 'gat':
        model = GAT(trainset.num_features, args.nhids, args.nheads).to(device)
    elif args.model == 'sage':
        model = SAGE(trainset.num_features, args.nhids).to(device)
    else:
        raise ValueError('Unknown model: {}'.format(args.model))

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = th.nn.BCELoss()

    total_train_loss = []
    total_test_loss = []

    total_train_accuracy = []
    total_test_accuracy = []

    conf_matrix = th.zeros(2, 2)

    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        model.train()
        batch_train_loss = []
        batch_train_accuracy = []
        for data in trainloader:
            optimizer.zero_grad()
            data = data.to(device)
        
            hn = model(data.edge_index, data.x)
            hg = readout(hn, data.batch, dim=0).view(-1)

            loss = loss_fn(hg, data.y.float())
            
            loss.backward()
            optimizer.step()
            batch_train_loss.append(loss.item())
            batch_train_accuracy.append(thm.accuracy(hg, data.y).item())

        total_train_loss.append(th.tensor(batch_train_loss).mean().item())
        total_train_accuracy.append(th.tensor(batch_train_accuracy).mean().item())

        model.eval()
        batch_test_loss = []
        batch_test_accuracy = []
        for data in testloader:
            data = data.to(device)
            hn = model(data.edge_index, data.x)
            hg = readout(hn, data.batch, dim=0).view(-1)
            loss = loss_fn(hg, data.y.float())
            batch_test_loss.append(loss.item()) 
            batch_test_accuracy.append(thm.accuracy(hg, data.y).item())

            conf_matrix = conf_matrix + thm.confusion_matrix(hg, data.y, num_classes=2)

        total_test_loss.append(th.tensor(batch_test_loss).mean().item())
        total_test_accuracy.append(th.tensor(batch_test_accuracy).mean().item())
        
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(args.epochs)), total_train_loss, label='training')
    plt.plot(list(range(args.epochs)), total_test_loss, label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'./figures/loss-{args.model}-{args.dataset}-{args.features}.png')


    plt.figure(figsize=(8, 6))
    plt.plot(list(range(args.epochs)), total_train_accuracy, label='training')
    plt.plot(list(range(args.epochs)), total_test_accuracy, label='test')
    plt.xlabel('epoch')
    plt.ylabel('performance')
    plt.legend()
    plt.savefig(f'./figures/accuracy-{args.model}-{args.dataset}-{args.features}.png')

    cmd = ConfusionMatrixDisplay(conf_matrix.numpy())

    cmd.plot()
    plt.savefig(f'./figures/confusion-matrix-{args.model}-{args.dataset}-{args.features}.png')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='politifact', choices=['politifact', 'gossipcop'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--features', type=str, default='bert', choices=['bert', 'spacy'])
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--nhids', type=int, default=32)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'sum'])
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage'])
    
    args = parser.parse_args()
    main(args)
