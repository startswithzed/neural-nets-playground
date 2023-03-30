import torch
import torch.nn.functional as F
import random
import argparse

words = open('../data/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
chtoi ={ch:i+1 for i, ch in enumerate(chars)} 
chtoi['.'] = 0 # separator token
itoch = {i:ch for ch, i in chtoi.items()} 

vocab_size = len(chars) + 1 

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--iters', help='Number of iterations (int)', default=10)
parser.add_argument('-l', '--lr', help='Learning rate (float)', default=0.1)
parser.add_argument('-s', '--num_samples', help='Number of samples (int)', default=10)
parser.add_argument('-c', '--len_context', help='Context length (int)', default=3)
parser.add_argument('-e', '--num_embed', help='Number of emdeddings (int)', default=2)
parser.add_argument('-n', '--num_hidden', help='Number of hidden layer neurons (int)', default=100)
parser.add_argument('-b', '--batch_size', help='Number of samples in a batch (int)', default=32)

args = parser.parse_args()

len_context = int(args.len_context)
n_embed = int(args.num_embed) 
n_hidden = int(args.num_hidden)
n_iters = int(args.iters)
batch_size = int(args.batch_size)
lr = float(args.lr)
n_samples = int(args.num_samples)

def build_dataset(words):
  X, Y = [], []

  for w in words:
    context = [0] * len_context 
    for ch in w + '.':
      ix = chtoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] 


  X = torch.tensor(X)
  Y = torch.tensor(Y)
    
  return X, Y

def get_splits(words):
  random.shuffle(words)

  n1 = int(0.8*len(words))
  n2 = int(0.9*len(words))

  Xtr, Ytr = build_dataset(words[:n1])
  Xval, Yval = build_dataset(words[n1:n2])
  Xtest, Ytest = build_dataset(words[n2:])

  return Xtr, Ytr, Xval, Yval, Xtest, Ytest

def init_params():
  C = torch.randn((vocab_size, n_embed))
  W1 = torch.randn((n_embed * len_context, n_hidden))
  B1 = torch.randn(n_hidden)
  W2 = torch.randn((n_hidden, vocab_size))
  B2 = torch.randn(vocab_size)
  
  params = [C, W1, B1, W2, B2]
  num_params = sum(p.nelement() for p in params)
  print(f'Total parameters in the network: {num_params}\n')

  for p in params:
      p.requires_grad = True

  return params

def train(n_iters, params, Xtr, Ytr, batch_size, lr, n_embed, block_size, print_loss=True):
  C = params[0]
  W1 = params[1]
  B1 = params[2]
  W2 = params[3]
  B2 = params[4]

  for i in range(n_iters):
    # create batches
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
    emb = C[Xtr[ix]]

    # forward pass
    h = torch.tanh(emb.view(-1, n_embed*block_size) @ W1 + B1)
    logits = h @ W2 + B2
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in params:
      p.grad = None
    loss.backward()
    
    #update
    for p in params:
      p.data += -lr * p.grad

    if print_loss:
      print(loss.item()) # batch-wise loss

  return params


def get_loss(params,  n_embed, block_size, X_split, Y_split):
  C = params[0]
  W1 = params[1]
  B1 = params[2]
  W2 = params[3]
  B2 = params[4]
  
  emb = C[X_split] 
  h = torch.tanh(emb.view(-1, n_embed*block_size) @ W1 + B1)
  logits = h @ W2 + B2
  loss = F.cross_entropy(logits, Y_split)
  
  return loss.item()

def get_samples(n_samples, params, block_size):
  C = params[0]
  W1 = params[1]
  B1 = params[2]
  W2 = params[3]
  B2 = params[4]

  print(f'Sample names:\n')

  for _ in range(n_samples):
    out = []
    context = [0] * block_size 
    while True:
        emb = C[torch.tensor([context])] 
        h = torch.tanh(emb.view(1, -1) @ W1 + B1)
        logits = h @ W2 + B2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
          break

    print(''.join(itoch[i] for i in out)) 

Xtr, Ytr, Xval, Yval, Xtest, Ytest = get_splits(words)
params = init_params()
params = train(n_iters, params, Xtr, Ytr, batch_size, lr, n_embed, len_context)
print(f'\nTraining Loss: {get_loss(params, n_embed, len_context, Xtr, Ytr)}')
print(f'Validation Loss: {get_loss(params, n_embed, len_context, Xval, Yval)}\n')
get_samples(n_samples, params, len_context)

