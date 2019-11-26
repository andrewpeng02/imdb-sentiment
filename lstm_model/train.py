import click
from pathlib import Path
import time

from lstm_model.model import IMDBLSTM
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lstm_model.dataset import IMDBDataset


@click.command()
@click.argument('num_epochs', type=int,  default=3)
@click.argument('seq_length', type=int,  default=196)
@click.argument('batch_size', type=int,  default=16)
@click.argument('lr', type=float,  default=1e-4)
@click.argument('batch_size', type=int,  default=16)
@click.argument('vocab_size', type=int,  default=5000)
@click.argument('embed_dim', type=int,  default=256)
@click.argument('hidden_dim', type=int,  default=256)
@click.argument('num_layers', type=int,  default=4)
@click.argument('lstm_dropout', type=float,  default=0.3)
@click.argument('fc_dropout', type=float,  default=0.3)
def main(**kwargs):
    project_path = str(Path(__file__).resolve().parents[1])
    train_dataset = IMDBDataset(project_path + '/data/train.csv', kwargs['seq_length'])
    train_loader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=True, num_workers=kwargs['batch_size'])
    valid_dataset = IMDBDataset(project_path + '/data/validation.csv', kwargs['seq_length'])
    valid_loader = DataLoader(valid_dataset, batch_size=kwargs['batch_size'], shuffle=True, num_workers=kwargs['batch_size'])

    model = IMDBLSTM(kwargs['vocab_size'], kwargs['embed_dim'], kwargs['hidden_dim'], kwargs['num_layers'],
                     kwargs['lstm_dropout'], kwargs['fc_dropout']).to('cuda')

    optim = Adam(model.parameters(), lr=kwargs['lr'])
    criterion = nn.CrossEntropyLoss()

    st = time.time()
    train(train_loader, valid_loader, model, optim, criterion, kwargs['num_epochs'], kwargs['batch_size'])
    print(f'Training time: {time.time() - st}sec')


def train(train_loader, valid_loader, model, optim, criterion, num_epochs, batch_size):
    model.train()

    for epoch in range(num_epochs):
        h = model.init_hidden(batch_size)
        total_loss = 0

        for step, (seqs, labels) in enumerate(iter(train_loader)):
            h = tuple([each.data for each in h])
            seqs = seqs.to('cuda')
            labels = labels.to('cuda').long()

            optim.zero_grad()
            outputs, h = model(seqs, h)
            loss = criterion(outputs, labels)

            loss.backward()
            optim.step()

            total_loss += loss.item()
            if step % 250 == 249:
                val_loss, val_acc = validate(valid_loader, model, criterion, batch_size)
                print(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \t '
                      f'Train Loss: {total_loss / 250} \t Val Loss: {val_loss} \t Val Acc: {val_acc}')
                total_loss = 0


def validate(valid_loader, model, criterion, batch_size):
    h = model.init_hidden(batch_size)
    model.eval()

    total_loss = 0
    total_acc = 0
    for seqs, labels in iter(valid_loader):
        with torch.no_grad():
            h = tuple([each.data for each in h])
            seqs = seqs.to('cuda')
            labels = labels.to('cuda').long()

            outputs, h = model(seqs, h)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)
    model.train()
    return total_loss / len(valid_loader), total_acc / len(valid_loader)


def accuracy(preds, labels):
    preds = torch.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    return torch.sum(preds == labels).item() / len(labels)


if __name__ == "__main__":
    main()
