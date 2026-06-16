import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from transformer_blocks import Block

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# ============================================================
# GANTI MODE DI SINI: "char", "word", atau "bpe"
# ============================================================
MODE = "word"
# ============================================================

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ── TOKENISASI ──────────────────────────────────────────────

if MODE == "char":
    print("\n[MODE] Character-level tokenization")
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda ids: ''.join([itos[i] for i in ids])
    ids = encode(text)

elif MODE == "word":
    print("\n[MODE] Word-level tokenization")
    words = text.split()
    vocab = sorted(set(words))
    vocab_size = len(vocab)
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}
    encode = lambda s: [stoi[w] for w in s.split() if w in stoi]
    decode = lambda ids: ' '.join([itos[i] for i in ids])
    ids = encode(text)

elif MODE == "bpe":
    print("\n[MODE] BPE tokenization (SentencePiece)")
    spm.SentencePieceTrainer.Train(
        input="corpus.txt",
        model_prefix="tokenizer",
        vocab_size=200,
        model_type="bpe"
    )
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer.model")
    ids = sp.encode(text, out_type=int)
    vocab_size = sp.get_piece_size()
    encode = lambda s: sp.encode(s, out_type=int)
    decode = lambda ids: sp.decode(ids)

print(f"Vocab size : {vocab_size}")
print(f"Total tokens: {len(ids)}")

data = torch.tensor(ids, dtype=torch.long)

# ── HYPERPARAMETER ───────────────────────────────────────────
block_size    = 32
embedding_dim = 64
n_heads       = 2
n_layers      = 2
lr            = 1e-3
epochs        = 3000

# ── BATCH ────────────────────────────────────────────────────
def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i+block_size]   for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ── MODEL ────────────────────────────────────────────────────
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs  = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# ── TRAINING ─────────────────────────────────────────────────
model     = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print(f"\nMemulai training [{MODE}] selama {epochs} steps...\n")
for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")

# ── GENERATE ─────────────────────────────────────────────────
print("\n" + "="*50)
print(f"Generated text [{MODE}]:")
print("="*50)

if MODE == "char":
    start = "counter"
    context = torch.tensor([encode(start)], dtype=torch.long)
elif MODE == "word":
    start = "Counter-Strike"
    context = torch.tensor([encode(start)], dtype=torch.long)
elif MODE == "bpe":
    start = "counter strike"
    context = torch.tensor([encode(start)], dtype=torch.long)

out = model.generate(context, max_new_tokens=100)
generated_ids = out[0].tolist()
print(decode(generated_ids))
print("="*50)