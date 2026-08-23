"""P3 pipeline: from-scratch Transformer encoder for text classification.

Attention, positional encoding, masking, and pooling are all implemented here
rather than pulled from ``torch.nn.TransformerEncoder``. That is deliberate: the
transformer-specific fault families (attention masking, score scaling, head
reshaping, positional encoding, pooling over padding) only form valid debugging
episodes if the faulty line is in source the agent can actually read and patch.

This is the CLEAN template; episode generators mutate a copy to inject one bug.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from silentml.pipelines.base import DataMeta, EvalMetrics, History

# --- Configuration -----------------------------------------------------------
CONFIG = {
    "lr": 3e-4,
    "batch_size": 32,
    "epochs": 6,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 256,
    "dropout": 0.1,
    "max_len": 64,
    "vocab_size": 512,
    "n_train": 3000,
    "n_val": 1000,
}

# Four classes over a controlled order-sensitive task (see _make_corpus): every
# document contains three of the four key words, and the label is whichever one
# appears EARLIEST. Presence is therefore uninformative - a bag-of-words model
# scores at chance - so the model must combine positional information with
# attention to select the first key word.
CATEGORIES = ["key_red", "key_blue", "key_green", "key_gold"]
KEY_WORDS = ["red", "blue", "green", "gold"]
N_KEYS_PER_DOC = 3

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2

import silentml  # noqa: E402

_DATA_ROOT = Path(
    os.environ.get("SILENTML_DATA", Path(silentml.__file__).resolve().parent.parent / ".data")
)


# --- Tokenisation ------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], vocab_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
            counts[tok] = counts.get(tok, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<cls>": CLS_ID}
    for tok, _ in ordered[: vocab_size - 3]:
        vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    """Encode to a fixed-length id sequence: [CLS] tokens... [PAD]..."""
    ids = [CLS_ID] + [vocab.get(tok, UNK_ID) for tok in tokenize(text)][: max_len - 1]
    return ids + [PAD_ID] * (max_len - len(ids))


# --- Model -------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return x + self.pe[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        # (batch, seq, d_model) -> (batch, n_heads, seq, d_head)
        return x.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_heads, seq, d_head = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, n_heads * d_head)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # pad_mask: (batch, seq) with True at real tokens, False at padding.
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Prevent attending to padding positions.
        mask = pad_mask.unsqueeze(1).unsqueeze(2)          # (batch, 1, 1, seq)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        return self.out_proj(self._merge_heads(context))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)
        self.ff_act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, pad_mask)))
        ff = self.ff_out(self.dropout(self.ff_act(self.ff_in(x))))
        x = self.norm2(x + self.dropout(ff))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        d_model = CONFIG["d_model"]
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoding = PositionalEncoding(d_model, CONFIG["max_len"])
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, CONFIG["n_heads"], CONFIG["d_ff"], CONFIG["dropout"])
            for _ in range(CONFIG["n_layers"])
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pad_mask = tokens != PAD_ID                       # (batch, seq)
        # nn.Embedding initialises to N(0, 1), which is already the same scale as
        # the sinusoidal encodings, so no sqrt(d_model) rescaling is applied here.
        x = self.embedding(tokens)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, pad_mask)

        # Classify from the [CLS] position, which attends over the whole sequence.
        pooled = x[:, 0, :]
        return self.classifier(pooled)


def build_model() -> nn.Module:
    return TransformerClassifier(CONFIG["vocab_size"], len(CATEGORIES))


# --- Data --------------------------------------------------------------------
_FILLER = [
    "the", "a", "of", "and", "to", "in", "with", "for", "on", "by", "from", "that",
    "system", "report", "value", "process", "record", "signal", "measure", "update",
    "note", "case", "level", "range", "state", "form", "unit", "phase", "result",
    "engine", "orbit", "harvest", "soil", "sensor", "module", "buffer", "channel",
]
def _make_corpus(n_samples: int, seed: int) -> tuple[list[str], list[int]]:
    """Generate the order-sensitive corpus deterministically.

    Each document is filler text into which three distinct key words are placed at
    distinct positions; the label is the key word occupying the earliest position.
    Because the same three key words are present regardless of the label, presence
    carries no information and the decision depends entirely on order. Documents
    vary in length, so padding (and therefore attention masking) is always
    exercised.
    """
    rng = torch.Generator().manual_seed(seed)

    def _pick(items: list[str], k: int) -> list[str]:
        idx = torch.randint(0, len(items), (k,), generator=rng).tolist()
        return [items[i] for i in idx]

    texts: list[str] = []
    labels: list[int] = []
    for _ in range(n_samples):
        length = int(torch.randint(16, 36, (1,), generator=rng).item())
        words = _pick(_FILLER, length)

        # Choose which key words appear, and at which (distinct, ordered) slots.
        keys = [KEY_WORDS[i] for i in
                torch.randperm(len(KEY_WORDS), generator=rng)[:N_KEYS_PER_DOC].tolist()]
        slots = sorted(torch.randperm(len(words), generator=rng)[:N_KEYS_PER_DOC].tolist())

        # Inserting from the back keeps earlier slot indices valid.
        for slot, word in sorted(zip(slots, keys), reverse=True):
            words.insert(slot, word)

        texts.append(" ".join(words))
        labels.append(KEY_WORDS.index(keys[0]))
    return texts, labels


class _Corpus:
    """Mimics the sklearn bunch interface used downstream."""

    def __init__(self, texts: list[str], labels: list[int]):
        self.data = texts
        self.target = labels


def _load_raw():
    # Disjoint seeds keep the validation split independent of training data.
    train = _Corpus(*_make_corpus(CONFIG["n_train"], seed=12345))
    test = _Corpus(*_make_corpus(CONFIG["n_val"], seed=54321))
    return train, test


def get_dataloaders(seed: int):
    train_raw, test_raw = _load_raw()

    g = torch.Generator().manual_seed(seed)
    tr_idx = torch.randperm(len(train_raw.data), generator=g)[: CONFIG["n_train"]].tolist()
    te_idx = torch.randperm(len(test_raw.data), generator=g)[: CONFIG["n_val"]].tolist()

    train_texts = [train_raw.data[i] for i in tr_idx]
    train_labels = [int(train_raw.target[i]) for i in tr_idx]
    val_texts = [test_raw.data[i] for i in te_idx]
    val_labels = [int(test_raw.target[i]) for i in te_idx]

    # Vocabulary is fit on training text only (no validation leakage).
    vocab = build_vocab(train_texts, CONFIG["vocab_size"])
    max_len = CONFIG["max_len"]

    x_tr = torch.tensor([encode(t, vocab, max_len) for t in train_texts], dtype=torch.long)
    y_tr = torch.tensor(train_labels, dtype=torch.long)
    x_va = torch.tensor([encode(t, vocab, max_len) for t in val_texts], dtype=torch.long)
    y_va = torch.tensor(val_labels, dtype=torch.long)

    loader_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr), batch_size=CONFIG["batch_size"], shuffle=True,
        generator=loader_g,
    )
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=128, shuffle=False)
    meta = DataMeta(
        num_classes=len(CATEGORIES), class_names=list(CATEGORIES),
        input_shape=(max_len,),
    )
    return train_loader, val_loader, meta


# --- Training ----------------------------------------------------------------
def _make_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    if config["optimizer"] == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
    return torch.optim.SGD(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, val_loader, config, seed) -> History:
    device = _device()
    model.to(device)
    optimizer = _make_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    hist = History(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    for _epoch in range(config["epochs"]):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for tokens, targets in train_loader:
            tokens, targets = tokens.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * tokens.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
        hist.train_loss.append(running_loss / total)
        hist.train_acc.append(correct / total)

        val = evaluate(model, val_loader)
        hist.val_loss.append(val.loss)
        hist.val_acc.append(val.accuracy)
    return hist


@torch.no_grad()
def evaluate(model, val_loader) -> EvalMetrics:
    device = _device()
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    for tokens, targets in val_loader:
        tokens, targets = tokens.to(device), targets.to(device)
        outputs = model(tokens)
        total_loss += criterion(outputs, targets).item() * tokens.size(0)
        preds = outputs.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        for t, p in zip(targets.tolist(), preds.tolist()):
            per_class_total[t] = per_class_total.get(t, 0) + 1
            per_class_correct[t] = per_class_correct.get(t, 0) + int(p == t)
    per_class_acc = {
        c: per_class_correct.get(c, 0) / per_class_total[c] for c in sorted(per_class_total)
    }
    return EvalMetrics(
        accuracy=correct / total, per_class_accuracy=per_class_acc, loss=total_loss / total
    )
