import click
from pathlib import Path
import time

from transformers import DistilBertForSequenceClassification
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
from distilbert_model.dataset import IMDBDataset


@click.command()
@click.argument('num_epochs', type=int,  default=3)
@click.argument('seq_length', type=int,  default=256)
@click.argument('batch_size', type=int,  default=16)
@click.argument('lr_transformer', type=float,  default=1e-5)
@click.argument('lr_classifier', type=float,  default=1e-5)
@click.argument('eps', type=float,  default=1e-8)
def main(**kwargs):
    project_path = str(Path(__file__).resolve().parents[1])
    train_dataset = IMDBDataset(project_path + '/data/train.csv', kwargs['seq_length'])
    train_loader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=True, num_workers=4)
    valid_dataset = IMDBDataset(project_path + '/data/validation.csv', kwargs['seq_length'])
    valid_loader = DataLoader(valid_dataset, batch_size=kwargs['batch_size'], shuffle=True, num_workers=4)

    print('Downloading the pretrained DistilBert model...')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to('cuda')
    lrs = [{'params': model.distilbert.parameters(), 'lr': kwargs['lr_transformer']},
           {'params': model.pre_classifier.parameters()},
           {'params': model.classifier.parameters()}]
    optim = Adam(lrs, lr=kwargs['lr_classifier'], eps=kwargs['eps'])

    print('Training...')
    st = time.time()
    train(train_loader, valid_loader, model, optim, kwargs['num_epochs'])
    print(f'Training time: {time.time() - st}sec')


def train(train_loader, valid_loader, model, optim, num_epochs):
    val_loss, val_acc = validate(valid_loader, model)
    print(f'Val Loss: {val_loss} \t Val Acc: {val_acc}')
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for step, (seqs, labels) in enumerate(iter(train_loader)):
            seqs = seqs.to('cuda')
            labels = labels.to('cuda').long()

            optim.zero_grad()
            loss = model(seqs, labels=labels)[0]

            loss.backward()
            optim.step()

            total_loss += loss.item()
            if step % 250 == 249:
                val_loss, val_acc = validate(valid_loader, model)
                print(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \t '
                      f'Train Loss: {total_loss / 250} \t Val Loss: {val_loss} \t Val Acc: {val_acc}')
                total_loss = 0


def validate(valid_loader, model):
    model.eval()

    total_loss = 0
    total_acc = 0
    for seqs, labels in iter(valid_loader):
        with torch.no_grad():
            seqs = seqs.to('cuda')
            labels = labels.to('cuda').long()

            loss, outputs = model(seqs, labels=labels)
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
