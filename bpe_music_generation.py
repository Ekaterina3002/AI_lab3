"""
Генерация музыки с BPE-токенизацией (Byte Pair Encoding).

Пайплайн:
  1. MIDI -> поток базовых музыкальных токенов (pitch + длительность + паузы).
  2. Обучение BPE поверх базовых токенов: частые соседние пары сливаются
     в новые "составные" токены (мотивы). Это data-driven альтернатива
     эвристической пословной токенизации (аккорды/фразы).
  3. Кодирование корпуса в id-последовательность.
  4. Обучение трёх генеративных моделей: RNN, LSTM, Transformer.
  5. Сравнение loss, график, генерация музыки, отчёт.

Запуск:  python bpe_music_generation.py
"""

import os
import glob
import json
import time
from collections import Counter

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Параметры эксперимента
# ----------------------------------------------------------------------------
NUM_FILES = 15          # сколько MIDI-файлов берём для обучения
MAX_BASE_TOKENS = 60_000    # ограничение на размер базового корпуса
NUM_MERGES = 300        # количество BPE-слияний (размер "subword" словаря)
REST_GAP = 0.5          # пауза (сек) -> токен <REST>
DUR_BUCKETS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # бакеты длительностей
SEQ_LENGTH = 64
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# 1. Базовая токенизация MIDI: pitch + длительность + паузы
# ----------------------------------------------------------------------------
def _dur_bucket(duration):
    """Квантование длительности ноты в дискретный бакет."""
    for b in DUR_BUCKETS:
        if duration <= b:
            return b
    return DUR_BUCKETS[-1]


def midi_to_base_tokens(midi_path):
    """MIDI -> список базовых токенов.

    Каждая нота кодируется парой токенов: высота 'P<pitch>' и длительность
    'D<bucket>'. Между нотами с большим зазором вставляется '<REST>'.
    Мелкий "алфавит" из ~100 базовых токенов -> у BPE есть что сливать.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return []

    tokens = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes = sorted(instrument.notes, key=lambda n: n.start)
        for i, note in enumerate(notes):
            dur = note.end - note.start
            tokens.append(f"P{note.pitch}")
            tokens.append(f"D{_dur_bucket(dur)}")
            if i + 1 < len(notes):
                gap = notes[i + 1].start - note.end
                if gap > REST_GAP:
                    tokens.append("<REST>")
    return tokens


# ----------------------------------------------------------------------------
# 2. BPE-токенизатор
# ----------------------------------------------------------------------------
class BPETokenizer:
    """Классический Byte Pair Encoding поверх базовых музыкальных токенов.

    Сегменты ограничиваются токеном <REST> (слияния не пересекают паузы),
    что аналогично границам слов в текстовом BPE.
    """

    SPECIALS = ["<PAD>", "<UNK>", "<START>", "<END>"]

    def __init__(self, num_merges=NUM_MERGES):
        self.num_merges = num_merges
        self.merges = []            # список слитых пар [(a, b), ...] по порядку
        self.token_to_idx = {}
        self.idx_to_token = {}

    # --- обучение ---
    def _split_segments(self, base_tokens):
        """Разбить плоский поток на сегменты по <REST> (REST сохраняется)."""
        segments, cur = [], []
        for t in base_tokens:
            if t == "<REST>":
                if cur:
                    segments.append(cur)
                    cur = []
                segments.append(["<REST>"])
            else:
                cur.append(t)
        if cur:
            segments.append(cur)
        return segments

    @staticmethod
    def _count_pairs(segments):
        pairs = Counter()
        for seg in segments:
            for a, b in zip(seg, seg[1:]):
                pairs[(a, b)] += 1
        return pairs

    @staticmethod
    def _merge_segment(seg, pair, new_token):
        a, b = pair
        out, i = [], 0
        while i < len(seg):
            if i + 1 < len(seg) and seg[i] == a and seg[i + 1] == b:
                out.append(new_token)
                i += 2
            else:
                out.append(seg[i])
                i += 1
        return out

    def fit(self, base_tokens):
        segments = self._split_segments(base_tokens)

        # базовый алфавит
        base_vocab = sorted({t for seg in segments for t in seg})

        print(f"  Базовый алфавит: {len(base_vocab)} токенов, "
              f"сегментов: {len(segments)}")

        # итеративные слияния
        for step in range(self.num_merges):
            pairs = self._count_pairs(segments)
            if not pairs:
                break
            best, freq = pairs.most_common(1)[0]
            if freq < 2:
                break
            new_token = best[0] + "|" + best[1]
            self.merges.append(best)
            segments = [self._merge_segment(seg, best, new_token)
                        for seg in segments]
            if (step + 1) % 50 == 0:
                print(f"  Слияние {step+1}/{self.num_merges}: "
                      f"{best} (freq={freq}) -> {new_token[:40]}")

        # финальный словарь: спецтокены + базовый алфавит + слитые токены
        all_tokens = list(self.SPECIALS) + base_vocab
        for a, b in self.merges:
            all_tokens.append(a + "|" + b)
        # уникализируем, сохраняя порядок
        seen, ordered = set(), []
        for t in all_tokens:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        self.token_to_idx = {t: i for i, t in enumerate(ordered)}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        print(f"  Итоговый BPE-словарь: {len(self.token_to_idx)} токенов "
              f"(слияний выполнено: {len(self.merges)})")

    # --- применение ---
    def _apply_merges(self, base_tokens):
        segments = self._split_segments(base_tokens)
        for pair in self.merges:
            new_token = pair[0] + "|" + pair[1]
            segments = [self._merge_segment(seg, pair, new_token)
                        for seg in segments]
        return [t for seg in segments for t in seg]

    def encode(self, base_tokens):
        merged = self._apply_merges(base_tokens)
        unk = self.token_to_idx["<UNK>"]
        return [self.token_to_idx.get(t, unk) for t in merged]

    def decode(self, indices):
        return [self.idx_to_token.get(i, "<UNK>") for i in indices]

    @property
    def vocab_size(self):
        return len(self.token_to_idx)

    def save(self, vocab_path="bpe_vocab.json", merges_path="bpe_merges.json"):
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_idx, f, ensure_ascii=False, indent=2)
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False)


# ----------------------------------------------------------------------------
# 3. Датасет
# ----------------------------------------------------------------------------
class MusicalDataset(Dataset):
    def __init__(self, sequences, seq_length=SEQ_LENGTH):
        self.sequences = sequences
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.sequences) - self.seq_length)

    def __getitem__(self, idx):
        x = self.sequences[idx:idx + self.seq_length]
        y = self.sequences[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ----------------------------------------------------------------------------
# 4. Модели
# ----------------------------------------------------------------------------
class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        out, _ = self.rnn(self.embedding(x))
        return self.fc(out)


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out)


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4,
                 num_layers=3, max_seq_len=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=embedding_dim * 4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        emb = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(emb, mask=mask)
        return self.fc(out)


# ----------------------------------------------------------------------------
# 5. Обучение и генерация
# ----------------------------------------------------------------------------
def train_model(model, loader, vocab_size, epochs=EPOCHS, lr=LR):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = <PAD>
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / max(1, len(loader))
        losses.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg:.4f}")
    return losses


def generate_music(model, tokenizer, length=200, temperature=0.8):
    model.eval()
    forbidden = {tokenizer.token_to_idx.get(t, 0)
                 for t in BPETokenizer.SPECIALS}
    # стартуем с реального (не спец-) токена
    start = next((i for t, i in tokenizer.token_to_idx.items()
                  if i not in forbidden), 4)
    current = [start]
    generated = []
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor(current[-SEQ_LENGTH:]).unsqueeze(0).to(DEVICE)
            logits = model(inp)[0, -1, :] / temperature
            for idx in forbidden:
                logits[idx] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            generated.append(nxt)
            current.append(nxt)
    return tokenizer.decode(generated)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BPE-ТОКЕНИЗАЦИЯ: генерация музыки (RNN / LSTM / Transformer)")
    print("=" * 60)
    print(f"Устройство: {DEVICE}")

    # данные
    midi_files = glob.glob("data/maestro-v3.0.0/**/*.mid", recursive=True)
    midi_files += glob.glob("data/maestro-v3.0.0/**/*.midi", recursive=True)
    midi_files = sorted(midi_files)
    if not midi_files:
        raise SystemExit("❌ MIDI-файлы не найдены в data/maestro-v3.0.0/")
    selected = midi_files[:NUM_FILES]
    print(f"Найдено MIDI: {len(midi_files)}, используем: {len(selected)}")

    # базовая токенизация
    print("\n[1] Базовая токенизация MIDI...")
    base_tokens = []
    for idx, path in enumerate(selected):
        toks = midi_to_base_tokens(path)
        base_tokens.extend(toks)
        if len(base_tokens) >= MAX_BASE_TOKENS:
            print(f"  Достигнут лимит {MAX_BASE_TOKENS} токенов на файле {idx+1}")
            break
    base_tokens = base_tokens[:MAX_BASE_TOKENS]
    print(f"  Всего базовых токенов: {len(base_tokens)}, "
          f"уникальных: {len(set(base_tokens))}")

    # обучение BPE
    print("\n[2] Обучение BPE...")
    t0 = time.time()
    tokenizer = BPETokenizer(num_merges=NUM_MERGES)
    tokenizer.fit(base_tokens)
    tokenizer.save()
    print(f"  BPE обучен за {time.time()-t0:.1f} c")

    # кодирование
    print("\n[3] Кодирование корпуса...")
    sequences = tokenizer.encode(base_tokens)
    compression = len(base_tokens) / max(1, len(sequences))
    print(f"  Длина id-последовательности: {len(sequences)} "
          f"(сжатие x{compression:.2f})")

    dataset = MusicalDataset(sequences, SEQ_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    vocab_size = tokenizer.vocab_size
    print(f"  Батчей: {len(loader)}, размер словаря: {vocab_size}")

    # обучение моделей
    print("\n[4] Обучение моделей...")
    model_defs = {
        "RNN": MusicRNN(vocab_size),
        "LSTM": MusicLSTM(vocab_size),
        "Transformer": MusicTransformer(vocab_size),
    }
    history, trained = {}, {}
    for name, model in model_defs.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()
        losses = train_model(model, loader, vocab_size)
        history[name] = losses
        trained[name] = model
        torch.save(model.state_dict(), f"model_{name.lower()}_bpe.pth")
        print(f"  {name} обучен за {time.time()-t0:.1f} c, "
              f"финальная loss: {losses[-1]:.4f}")

    # график
    print("\n[5] График сравнения...")
    plt.figure(figsize=(12, 6))
    colors = {"RNN": "blue", "LSTM": "green", "Transformer": "red"}
    for name, losses in history.items():
        plt.plot(losses, label=name, color=colors[name], linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Сравнение моделей (BPE-токенизация музыки)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("training_comparison_bpe.png", dpi=150, bbox_inches="tight")
    print("  Сохранён training_comparison_bpe.png")

    # генерация
    print("\n[6] Генерация музыки...")
    for name, model in trained.items():
        for temp in [0.6, 0.8, 1.0]:
            gen = generate_music(model, tokenizer, length=150, temperature=temp)
            fname = f"generated_{name.lower()}_bpe_temp{temp}.txt"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(" ".join(gen))
        print(f"  {name}: сэмплы сохранены (temp 0.6/0.8/1.0)")

    # отчёт
    print("\n[7] Отчёт...")
    best = min(history.items(), key=lambda kv: min(kv[1]))[0]
    lines = [
        "# Отчёт: генеративные сети с BPE-токенизацией\n",
        "## Параметры эксперимента",
        f"- Тип токенизации: **BPE (Byte Pair Encoding)**",
        f"- MIDI-файлов: {len(selected)}",
        f"- Базовых токенов: {len(base_tokens)}",
        f"- BPE-слияний: {len(tokenizer.merges)}",
        f"- Размер словаря: {vocab_size}",
        f"- Сжатие BPE: x{compression:.2f}",
        f"- Длина последовательности: {SEQ_LENGTH}",
        f"- Эпох: {EPOCHS}",
        f"- Устройство: {DEVICE}\n",
        "## Результаты обучения\n",
        "| Модель | Финальная Loss | Лучшая Loss |",
        "|--------|----------------|-------------|",
    ]
    for name, losses in history.items():
        lines.append(f"| {name} | {losses[-1]:.4f} | {min(losses):.4f} |")
    lines += [
        "",
        "## Выводы",
        f"- **Лучшая модель по loss**: {best}",
        "- BPE снижает длину последовательности за счёт слияния частых "
        "музыкальных мотивов, что ускоряет обучение и расширяет контекст.",
        "",
        "## Файлы",
        "- model_rnn_bpe.pth, model_lstm_bpe.pth, model_transformer_bpe.pth",
        "- bpe_vocab.json, bpe_merges.json",
        "- generated_*_bpe_temp*.txt",
        "- training_comparison_bpe.png",
    ]
    with open("REPORT_BPE.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  Сохранён REPORT_BPE.md")

    print("\n" + "=" * 60)
    print("✅ Готово!")
    print("Финальные loss:")
    for name, losses in history.items():
        print(f"  {name:12} | финальная: {losses[-1]:.4f} | лучшая: {min(losses):.4f}")


if __name__ == "__main__":
    main()
