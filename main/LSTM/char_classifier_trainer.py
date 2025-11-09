import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# ==================== ПАРАМЕТРЫ ====================
dataname = input("Filename: ")
CSV_FILE = f'data/{dataname}.csv'  # Путь к CSV файлу
MAX_VOCAB_SIZE = 60000  # Максимальный размер словаря
MAX_SEQ_LENGTH = 750  # Максимальная длина последовательности
EMBEDDING_DIM = 128  # Размерность эмбеддингов
HIDDEN_DIM = 256  # Размерность скрытого слоя LSTM
NUM_LAYERS = 2  # Количество слоёв LSTM
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 3
TEST_SIZE = 0.2  # Доля данных для валидации
MIN_SAMPLES = 500 # Минимальное количество примеров на класс
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = f'{dataname}.pth'
VOCAB_SAVE_PATH = f'{dataname}_vocab.pth'
# ===================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class Vocabulary:
    """Класс для работы со словарём токенов"""
    def __init__(self, max_size=None):
        self.max_size = max_size
        self.token2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2token = {0: '<PAD>', 1: '<UNK>'}
        self.token_counts = Counter()
        
    def build_vocab(self, texts):
        """Построение словаря на основе текстов"""
        print("Построение словаря...")
        for text in tqdm(texts):
            self.token_counts.update(text)
        
        # Берём наиболее частые токены
        most_common = self.token_counts.most_common(self.max_size - 2 if self.max_size else None)
        
        for idx, (token, _) in enumerate(most_common, start=2):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        
        print(f"Размер словаря: {len(self.token2idx)} токенов")
        print(f"Всего уникальных символов в текстах: {len(self.token_counts)}")
        
    def encode(self, text):
        """Преобразование текста в индексы"""
        return [self.token2idx.get(char, self.token2idx['<UNK>']) for char in text]
    
    def __len__(self):
        return len(self.token2idx)


class TextDataset(Dataset):
    """Датасет для текстовых данных"""
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx][:self.max_length]  # Обрезаем до максимальной длины
        encoded = self.vocab.encode(text)
        return torch.tensor(encoded, dtype=torch.long), self.labels[idx]


def collate_fn(batch):
    """Функция для создания батча с паддингом по максимальной длине в батче"""
    texts, labels = zip(*batch)
    
    # Находим максимальную длину в текущем батче
    max_len = max(len(text) for text in texts)
    
    # Паддинг до максимальной длины батча
    padded_texts = []
    lengths = []
    for text in texts:
        length = len(text)
        lengths.append(length)
        padded = torch.cat([text, torch.zeros(max_len - length, dtype=torch.long)])
        padded_texts.append(padded)
    
    return torch.stack(padded_texts), torch.tensor(labels, dtype=torch.long), torch.tensor(lengths)


class TextClassifier(nn.Module):
    """LSTM-классификатор текстов"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 для bidirectional
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        
        # Pack padded sequence для эффективной обработки
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Используем последние скрытые состояния обоих направлений
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output


def load_data(csv_path, min_samples_per_class=500):
    """Загрузка данных из CSV"""
    print(f"Загрузка данных из {csv_path}...")
    df = pd.read_csv(csv_path, sep='\t', header=None, names=['label', 'text'], 
                     on_bad_lines='skip', encoding='utf-8')
    
    # Преобразуем всё в строки и удаляем NaN
    df['label'] = df['label'].astype(str)
    df['text'] = df['text'].astype(str)
    
    # Удаляем строки с пустыми или nan текстами и метками
    df = df[df['text'].str.strip() != '']
    df = df[df['text'] != 'nan']
    df = df[df['label'].str.strip() != '']
    df = df[df['label'] != 'nan']
    
    print(f"Загружено {len(df)} записей (до фильтрации)")
    
    # Подсчитываем количество примеров для каждого класса
    class_counts = df['label'].value_counts()
    
    # Оставляем только классы с достаточным количеством примеров
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df = df[df['label'].isin(valid_classes)]
    
    excluded_classes = len(class_counts) - len(valid_classes)
    print(f"Исключено {excluded_classes} классов с менее чем {min_samples_per_class} примерами")
    print(f"Осталось {len(df)} записей в {len(valid_classes)} классах")
    print(f"\nРаспределение классов:\n{df['label'].value_counts()}")
    
    return df['text'].tolist(), df['label'].tolist()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for texts, labels, lengths in tqdm(dataloader, desc='Training'):
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Валидация модели"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, labels, lengths in tqdm(dataloader, desc='Validation'):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    print(f"Используется устройство: {DEVICE}")
    
    # Загрузка данных
    texts, labels = load_data(CSV_FILE, min_samples_per_class=MIN_SAMPLES)
    
    # Преобразование меток в числовые значения
    unique_labels = sorted(list(set(labels)))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label2idx[label] for label in labels]
    num_classes = len(unique_labels)
    
    print(f"\nКоличество классов: {num_classes}")
    print(f"Метки: {unique_labels}")
    
    # Разделение на train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    
    print(f"\nРазмер обучающей выборки: {len(train_texts)}")
    print(f"Размер валидационной выборки: {len(val_texts)}")
    
    # Построение словаря
    vocab = Vocabulary(max_size=MAX_VOCAB_SIZE)
    vocab.build_vocab(train_texts)
    
    # Создание датасетов
    train_dataset = TextDataset(train_texts, train_labels, vocab, MAX_SEQ_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, vocab, MAX_SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, collate_fn=collate_fn)
    
    # Создание модели
    model = TextClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    print(f"\nАрхитектура модели:")
    print(model)
    print(f"\nВсего параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение
    best_val_acc = 0
    print("\n" + "="*50)
    print("Начало обучения")
    print("="*50)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label2idx': label2idx,
                'vocab_size': len(vocab),
                'num_classes': num_classes,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'vocab_token2idx': vocab.token2idx,
                'vocab_idx2token': vocab.idx2token
            }, MODEL_SAVE_PATH)
            print(f"✓ Модель сохранена (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*50)
    print(f"Обучение завершено!")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")
    print(f"Модель сохранена в: {MODEL_SAVE_PATH}")
    #print(f"Словарь сохранён в: {VOCAB_SAVE_PATH}")
    print("="*50)


if __name__ == '__main__':
    main()