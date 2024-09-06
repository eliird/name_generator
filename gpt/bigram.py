import torch
import torch.nn as nn
from torch.nn import functional as F

# hyper parametes
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


torch.manual_seed(1337)

with open('shakespear.txt', 'r') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])


# train and val splits 

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading

def get_batch(split: str):
    data = train_data if split== 'train' else val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return (x, y)


# defining the model
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        if targets is not None:
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return (logits.view(B, T, C), loss)
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def train():
    train_loss = []
    eval_loss = []
    for steps in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss =m(xb.to(device), yb.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # tracking stats
        train_loss.append(loss.item())
        
        if steps % eval_interval == 0:
            with torch.no_grad():
                m.eval()
                for e_step in range(eval_iters):
                    x_e, y_e = get_batch("test")
                    _, e_loss = m(x_e, y_e)
                    eval_loss.append(e_loss.item())
            m.train()   
            print(f"Train Step = {steps} | Validation Loss = {sum(eval_loss[-eval_iters:])/eval_iters}")
    return (train_loss, eval_loss)
        
# declaring the model
m = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

t_loss, e_loss = train()

t_loss =torch.tensor(t_loss).view(-1, int(len(t_loss)/100)).mean(dim=1)
e_loss =torch.tensor(e_loss).view(-1, int(len(e_loss)/100)).mean(dim=1)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(t_loss)
plt.title("Training Loss")
plt.savefig('training_bigram.jpg')

plt.figure()
plt.plot(e_loss)
plt.title("Validation Loss")
plt.savefig('validation_bigram.jpg')

