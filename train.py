import argparse # bib utilisé pour  faire liste des arguments 
#
import os  # utilisé pour lire path shemin
import torch as th 
import pandas as pd
import torchmetrics.functional as thm
from tqdm import tqdm # bib pour affiché le temps d'entrainement 
from models import LSTM, GRU
from utils import load_vocab, text_to_tokens 
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt # visualisation des plots 


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main(args):
    device = th.device('cpu')
    df = pd.read_json(os.path.join('../data', args.dataset + ".json"), orient='records', lines=True) # lire fichier json qui contient le dataset
    vocab = load_vocab(os.path.join('../data', args.dataset + ".vocab.json")) # load vocab est une fonction qui trouve dans le fichier utils qui lire le fichier vocab.json 
    
    trainset = df.sample(frac=0.8) 

    testset = df.drop(trainset.index)

    vocab_size = len(vocab)
    
    if args.model == 'lstm':
        model = LSTM(vocab_size).to(device)
    elif args.model == 'gru':
        model = GRU(vocab_size).to(device)
    else:
        raise ValueError("Invalid model")

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr) # pour optimiser l'erreur 
    criterion = th.nn.BCELoss() # (Binary Cross-Entropy Loss ) fonction loss entre y et y predit 
    
    model.train()

    total_train_loss = []
    total_test_loss = []

    total_train_accuracy = []
    total_test_accuracy = []

    predictions = []
    labels = []
    conf_matrix = th.zeros(2, 2)

    for _ in tqdm(range(10), desc='Epoch', leave=False):
        train_labels = []
        train_preds = []

        for idx, row in tqdm(trainset.iterrows(), total=len(trainset), desc='Training', leave=False):
            seq = text_to_tokens(row['text'], vocab) # fonction dans le fichier utils quit prend le text et les vocab de text
            if seq.nelement() == 0:
                continue
            label = th.tensor(row['label']).unsqueeze(0).float() #list true label 
            out = model(seq)
            loss = criterion(out, label)
            loss.backward()

            train_labels.append(label.item())
            train_preds.append(out.item())

        if idx % 10 == 0:
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss.append(loss.item())
            total_train_accuracy.append(
                thm.accuracy(th.tensor(train_preds), th.tensor(train_labels)).item()
            )
            train_preds = []
            train_labels = []
        


    model.eval()

    for idx, row in tqdm(testset.iterrows(), total=len(testset), desc='Testing'):
        seq = text_to_tokens(row['text'], vocab)
        if seq.nelement() == 0:
            continue
        label = th.tensor(row['label']).unsqueeze(0).float()
        out = model(seq)
        loss = criterion(out, label)

        total_train_loss.append(loss.item())

        predictions.append(out.item())
        labels.append(label.item())

        if idx % 10 == 0:
            total_test_accuracy.append(
                thm.accuracy(th.tensor(predictions), th.tensor(labels)).item()
            )
            
            total_test_loss.append(loss.item())

            conf_matrix = conf_matrix + thm.confusion_matrix(th.tensor(predictions), th.tensor(labels))


    plt.figure(figsize=(8, 6))
    plt.plot(list(range(args.epochs)), total_train_loss, label='training')
    plt.plot(list(range(args.epochs)), total_test_loss, label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'./figures/loss-{args.model}-{args.dataset}.png')


    plt.figure(figsize=(8, 6))
    plt.plot(list(range(args.epochs)), total_train_accuracy, label='training')
    plt.plot(list(range(args.epochs)), total_test_accuracy, label='test')
    plt.xlabel('epoch')
    plt.ylabel('performance')
    plt.legend()
    plt.savefig(f'./figures/accuracy-{args.model}-{args.dataset}.png')

    cmd = ConfusionMatrixDisplay(conf_matrix.numpy())

    cmd.plot()
    plt.savefig(f'./figures/confusion-matrix-{args.model}-{args.dataset}.png')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['politifact', 'gossipcop'], default='politifact')
    parser.add_argument('--model', type=str, choices=['lstm', 'gru'], default='lstm')
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    main(args)
