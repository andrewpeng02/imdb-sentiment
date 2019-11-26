import click
from pathlib import Path
import time

from transformer_model.model import IMDBTransformer
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_model.dataset import IMDBDataset


@click.command()
@click.argument('num_epochs', type=int,  default=5)
@click.argument('seq_length', type=int,  default=196)
@click.argument('batch_size', type=int,  default=16)
@click.argument('lr', type=float,  default=1e-5)
@click.argument('eps', type=float,  default=1e-9)
@click.argument('batch_size', type=int,  default=16)
@click.argument('vocab_size', type=int,  default=5000)
@click.argument('d_model', type=int,  default=256)
@click.argument('nhead', type=int,  default=8)
@click.argument('num_layers', type=int,  default=6)
@click.argument('pos_dropout', type=float,  default=0.2)
@click.argument('fc_dropout', type=float,  default=0.2)
def main(**kwargs):
    project_path = str(Path(__file__).resolve().parents[1])
    train_dataset = IMDBDataset(project_path + '/data/train.csv', kwargs['seq_length'])
    train_loader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=True, num_workers=kwargs['batch_size'])
    valid_dataset = IMDBDataset(project_path + '/data/validation.csv', kwargs['seq_length'])
    valid_loader = DataLoader(valid_dataset, batch_size=kwargs['batch_size'], shuffle=True, num_workers=kwargs['batch_size'])

    model = IMDBTransformer(kwargs['vocab_size'], kwargs['d_model'], kwargs['nhead'], kwargs['num_layers'],
                            kwargs['seq_length'], kwargs['pos_dropout'], kwargs['fc_dropout']).to('cuda')

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optim = Adam(model.parameters(), lr=kwargs['lr'], betas=(0.9, 0.98), eps=kwargs['eps'])
    criterion = nn.CrossEntropyLoss()

    st = time.time()
    train(train_loader, valid_loader, model, optim, criterion, kwargs['num_epochs'])
    print(f'Training time: {time.time() - st}sec')


def train(train_loader, valid_loader, model, optim, criterion, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for step, (seqs, masks, labels) in enumerate(iter(train_loader)):
            seqs = seqs.to('cuda')
            masks = masks.to('cuda')
            labels = labels.to('cuda').long()

            optim.zero_grad()
            outputs = model(seqs, masks)
            loss = criterion(outputs, labels)

            loss.backward()
            optim.step()

            total_loss += loss.item()
            if step % 250 == 249:
                val_loss, val_acc = validate(valid_loader, model, criterion)
                print(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \t '
                      f'Train Loss: {total_loss / 250} \t Val Loss: {val_loss} \t Val Acc: {val_acc}')
                total_loss = 0


def validate(valid_loader, model, criterion):
    model.eval()

    total_loss = 0
    total_acc = 0
    for seqs, masks, labels in iter(valid_loader):
        with torch.no_grad():
            seqs = seqs.to('cuda')
            masks = masks.to('cuda')
            labels = labels.to('cuda').long()

            outputs = model(seqs, masks)
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
