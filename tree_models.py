import os
import argparse
import torch as th
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main(args):
    trainset = UPFD(root='../data', name=args.dataset, feature=args.features, split='train')
    testset = UPFD(root='../data', name=args.dataset, feature=args.features, split='test')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    readout = scatter_add
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for data in trainloader:
        
        hg = readout(data.x, data.batch, dim=0).numpy()
        y = data.y.numpy()

        x_train.append(hg)
        y_train.append(y)

    for data in testloader:
        hg = readout(data.x, data.batch, dim=0).numpy()
        y = data.y.numpy()

        x_test.append(hg)
        y_test.append(y)

    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_train = np.concatenate(y_train, axis=0).flatten()
    y_test = np.concatenate(y_test, axis=0).flatten()

    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm)

    cmd.plot()
    plt.savefig('figures/confusion-matrix-random-forest-{args.dataset}-{args.features}.png')

    test_auc = metrics.roc_auc_score(y_test, y_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    test_recall = metrics.recall_score(y_test, y_pred)
    test_precision = metrics.precision_score(y_test, y_pred)
    test_micro_f1 = metrics.f1_score(y_test, y_pred, average='micro')
    test_macro_f1 = metrics.f1_score(y_test, y_pred, average='macro')


    """
    with open('results.txt', 'a') as f:
        f.write('==== {} - {} - {} (features only) ==== \n'.format('Random Forest', args.features, args.dataset))
        f.write('Test AUROC: {:.2f} %\n'.format(test_auc * 100))
        f.write('Test Accuracy: {:.2f} %\n'.format(test_accuracy * 100))
        f.write('Test Recall: {:.2f} %\n'.format(test_recall * 100))
        f.write('Test Precision: {:.2f} %\n'.format(test_precision * 100))
        f.write('Test Micro F1: {:.2f} %\n'.format(test_micro_f1 * 100))
        f.write('Test Macro F1: {:.2f} %\n'.format(test_macro_f1 * 100))
        
    f.close()
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='politifact', choices=['politifact', 'gossipcop'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--features', type=str, default='bert', choices=['bert', 'spacy'])
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    main(args)