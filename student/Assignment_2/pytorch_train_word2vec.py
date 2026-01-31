import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512 
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5 

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, pkl_path="processed_data.pkl"):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        df = d["skipgram_df"]
        self.centers = torch.tensor(df["center"].values, dtype=torch.long)
        self.contexts = torch.tensor(df["context"].values, dtype=torch.long)

        self.word2idx = d["word2idx"]
        self.idx2word = d["idx2word"]
        self.counter = d["counter"]

    def __len__(self):
        return len(self.centers)
    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    
# Load processed data
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_ids, context_ids):
        center_vecs = self.in_embed(center_ids)    
        context_vecs = self.out_embed(context_ids)
        scores = torch.sum(center_vecs * context_vecs, dim=1)
        return scores


# Precompute negative sampling distribution below
def build_negative_sampling_probs(counter, word2idx, power=0.75):
    vocab_size = len(word2idx)
    counts = torch.zeros(vocab_size, dtype=torch.float)

    for word, i in word2idx.items():
        counts[i] = float(counter.get(word, 0))
        
    counts = torch.clamp(counts, min=1.0)

    probs = counts.pow(power)
    probs = probs / probs.sum()

    return probs


# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset and DataLoader
ds = SkipGramDataset("processed_data.pkl")
vocab_size = len(ds.word2idx)

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Negative sampling distribution
negative_probs = build_negative_sampling_probs(ds.counter, ds.word2idx).to(device)


# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0

    for centers, contexts in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        centers = centers.to(device)   
        contexts = contexts.to(device)  
        B = centers.size(0)

        positive_scores = model(centers, contexts)         
        positive_labels = torch.ones_like(positive_scores)     
        positive_loss = loss_fn(positive_scores, positive_labels)
       
        negative_contexts = torch.multinomial(
            negative_probs,
            num_samples=B * NEGATIVE_SAMPLES,
            replacement=True
        ).view(B, NEGATIVE_SAMPLES)    
        mask = negative_contexts.eq(contexts.unsqueeze(1))
        
        while mask.any():
            negative_contexts[mask] = torch.multinomial(
            negative_probs,
            num_samples=int(mask.sum().item()),
            replacement=True)
            mask = negative_contexts.eq(contexts.unsqueeze(1))

        neg_scores = model(
            centers.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES).reshape(-1), 
            negative_contexts.reshape(-1)                                    
        )
        negative_labels = torch.zeros_like(neg_scores)  
        negative_loss = loss_fn(neg_scores, negative_labels)

        loss = positive_loss + negative_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

# Save embeddings and mappings
embeddings = model.in_embed.weight.detach().cpu().numpy()

with open("word2vec_embeddings.pkl", "wb") as f:
    pickle.dump( {"embeddings": embeddings, "word2idx": ds.word2idx, "idx2word": ds.idx2word}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
