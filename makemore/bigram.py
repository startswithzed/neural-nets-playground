import torch
import torch.nn.functional as F
import argparse


words = open('../data/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
chtoi = {ch: i+1 for i, ch in enumerate(chars)}
chtoi['.'] = 0
itoch = {i: ch for ch, i in chtoi.items()}

parser = argparse.ArgumentParser()

parser.add_argument(
    '-i', '--iters', help='Number of iterations (int)', default=10)
parser.add_argument('-l', '--lr', help='Learning Rate (float)', default=1)
parser.add_argument('-s', '--samples',
                    help='Number of Samples (int)', default=10)
parser.add_argument('-r', '--reg',
                    help='Regulaize Loss (bool)', default=True)
parser.add_argument(
    '-p', '--printw', help='Show weights (bool)', default=False)

args = parser.parse_args()

n_iters = int(args.iters)
lr = float(args.lr)
n_samples = int(args.samples)
regularize_loss = bool(args.reg)
show_weights = bool(args.printw)


def get_datasets(words):
    inputs, targets = [], []

    for w in words:
        chars = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = chtoi[ch1]
            ix2 = chtoi[ch2]
            inputs.append(ix1)
            targets.append(ix2)

    return torch.tensor(inputs), torch.tensor(targets)


def train(n_iters, inputs, targets):
    w = torch.randn((27, 27), requires_grad=True)
    inputs_enc = F.one_hot(inputs, num_classes=27).float()

    for _ in range(n_iters):
        # forward pass
        logits = inputs_enc @ w
        probs = torch.softmax(logits, dim=1)
        if regularize_loss:
            loss = - probs[torch.arange(probs.size(dim=0)),
                           targets].log().mean() + 0.01*(w**2).mean()
        else:
            loss = - probs[torch.arange(probs.size(dim=0)),
                           targets].log().mean()
        print(loss.item())

        # backward pass
        w.grad = None
        loss.backward()
        w.data += -lr * w.grad

        if show_weights:
            print(f'first row of the weight matrix is:\n{w[:1]}\n')

    return w


def get_samples(n_samples, probs):
    for i in range(n_samples):
        samples = []
        idx = 0

        while True:
            inputs_enc = F.one_hot(torch.tensor([idx]), num_classes=27).float()
            logits = inputs_enc @ w
            probs = torch.softmax(logits, dim=1)
            idx = torch.multinomial(
                probs, num_samples=1, replacement=True).item()
            samples.append(itoch[idx])
            if idx == 0:
                break
        print(''.join(samples))


inputs, targets = get_datasets(words)
w = train(n_iters, inputs, targets)
get_samples(n_samples, w)
