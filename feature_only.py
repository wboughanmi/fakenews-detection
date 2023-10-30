import os
import argparse
import torch as th
import torch.nn as nn
import torchmetrics.functional as thm
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main(args):
    device = th.device('cpu')
    trainset = UPFD(root='../data', name=args.dataset, feature=args.features, split='train')
    testset = UPFD(root='../data', name=args.dataset, feature=args.features, split='test')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    readout = scatter_add

    predictor = nn.Sequential(
        nn.Linear(trainset.num_features, 1),
        nn.Sigmoid()
    ).to(device)

    optimizer = th.optim.Adam(predictor.parameters(), lr=args.lr)
    loss_fn = th.nn.BCELoss()

    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        predictor.train()
        for data in trainloader:
            optimizer.zero_grad()
            data = data.to(device)
        
            hg = readout(data.x, data.batch, dim=0)
            y_pred = predictor(hg).view(-1)

            loss = loss_fn(y_pred, data.y.float())
            
            loss.backward()
            optimizer.step()
    
    predictor.eval()
    total_loss = []
    total_auc = []
    conf_matrix = th.zeros(2, 2)

    for data in testloader:
        data = data.to(device)
        hg = readout(data.x, data.batch, dim=0)
        y_pred = predictor(hg).view(-1)

        loss = loss_fn(y_pred, data.y.float())
        
        total_loss.append(loss.item()) 
        total_auc.append(thm.auroc(y_pred, data.y))


    test_loss = sum(total_loss) / len(total_loss)
    test_auc = sum(total_auc) / len(total_auc)

    print('Test loss: {:.4f}'.format(test_loss))
    print('Test AUC: {:.4f}'.format(test_auc))  

    # with open('results.txt', 'a') as f:
    #    f.write('==== {} - {} (features only) ===='.format(args.features, args.dataset))
    #    f.write('Test AUROC: {:.2f} %\n'.format(test_auc * 100))
    # f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='politifact', choices=['politifact', 'gossipcop'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--features', type=str, default='bert', choices=['bert', 'spacy'])
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    main(args)
