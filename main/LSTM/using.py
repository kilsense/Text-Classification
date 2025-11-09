import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import pandas as pd

# ==================== ПАРАМЕТРЫ ====================
MODEL_PATH = 'whole.pth'
MAX_SEQ_LENGTH = 750
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===================================================


class Vocabulary:
    """Класс для работы со словарём токенов"""
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        
    def encode(self, text):
        """Преобразование текста в индексы"""
        return [self.token2idx.get(char, self.token2idx.get('<UNK>', 1)) for char in text]
    
    def load_from_dict(self, token2idx, idx2token):
        """Загрузка словаря из словарей"""
        self.token2idx = token2idx
        self.idx2token = idx2token


class TextClassifier(nn.Module):
    """LSTM-классификатор текстов"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output


class TextClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Классификатор текстов")
        self.root.geometry("900x700")
        
        # Загрузка модели
        self.model = None
        self.vocab = None
        self.label2idx = None
        self.idx2label = None
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        """Создание интерфейса"""
        # Стиль
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
        
        # Верхняя панель с информацией
        info_frame = ttk.Frame(self.root, padding=10)
        info_frame.pack(fill='x')
        
        self.status_label = ttk.Label(info_frame, text="Загрузка модели...", 
                                      style='Info.TLabel', foreground='orange')
        self.status_label.pack(side='left')
        
        model_frame = ttk.Frame(self.root, padding=10)
        model_frame.pack(side='right')

        self.mname_entry = ttk.Entry(model_frame)
        self.mname_entry.pack(side='top')

        self.mname_entry.insert(0, "whole")

        self.mname_button = ttk.Button(model_frame, text="Загрузить", command=self.load_model)
        self.mname_button.pack(side='bottom')

        self.device_label = ttk.Label(info_frame, text=f"Устройство: {DEVICE}", 
                                      style='Info.TLabel')
        self.device_label.pack(side='right')
        
        # Notebook для вкладок
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Вкладка 1: Одиночная классификация
        self.single_frame = ttk.Frame(notebook, padding=10)
        notebook.add(self.single_frame, text='Одиночная классификация')
        self.setup_single_tab()
        
        # Вкладка 2: Пакетная классификация
        self.batch_frame = ttk.Frame(notebook, padding=10)
        notebook.add(self.batch_frame, text='Пакетная классификация')
        self.setup_batch_tab()
        
    def setup_single_tab(self):
        """Настройка вкладки одиночной классификации"""
        # Заголовок
        ttk.Label(self.single_frame, text="Введите текст для классификации:", 
                  style='Header.TLabel').pack(anchor='w', pady=(0, 5))
        
        # Текстовое поле ввода
        self.text_input = scrolledtext.ScrolledText(
            self.single_frame, height=10, font=('Arial', 11), wrap='word'
        )
        self.text_input.pack(fill='both', expand=True, pady=(0, 10))
        
        # Кнопки
        button_frame = ttk.Frame(self.single_frame)
        button_frame.pack(fill='x', pady=(0, 10))
        
        self.classify_btn = ttk.Button(
            button_frame, text="Классифицировать", 
            command=self.classify_single, state='disabled'
        )
        self.classify_btn.pack(side='left', padx=(0, 5))
        
        ttk.Button(button_frame, text="Очистить", 
                   command=self.clear_single).pack(side='left')
        
        # Результаты
        ttk.Label(self.single_frame, text="Результаты:", 
                  style='Header.TLabel').pack(anchor='w', pady=(10, 5))
        
        # Фрейм с результатами
        result_frame = ttk.Frame(self.single_frame)
        result_frame.pack(fill='both', expand=True)
        
        # Таблица результатов
        columns = ('Класс', 'Вероятность')
        self.result_tree = ttk.Treeview(result_frame, columns=columns, 
                                        show='headings', height=10)
        
        self.result_tree.heading('Класс', text='Класс')
        self.result_tree.heading('Вероятность', text='Вероятность (%)')
        self.result_tree.column('Класс', width=200)
        self.result_tree.column('Вероятность', width=150)
        
        scrollbar = ttk.Scrollbar(result_frame, orient='vertical', 
                                 command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        
        self.result_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
    def setup_batch_tab(self):
        """Настройка вкладки пакетной классификации"""
        # Инструкции
        ttk.Label(self.batch_frame, 
                  text="Загрузите CSV файл с текстами для классификации\n" +
                       "Формат: одна колонка с текстами (без заголовка) или две колонки (класс, текст)",
                  style='Info.TLabel').pack(anchor='w', pady=(0, 10))
        
        # Выбор файла
        file_frame = ttk.Frame(self.batch_frame)
        file_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(file_frame, text="Файл:").pack(side='left', padx=(0, 5))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, 
                              width=50, state='readonly')
        file_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        ttk.Button(file_frame, text="Выбрать файл", 
                   command=self.select_file).pack(side='left', padx=(0, 5))
        
        self.batch_classify_btn = ttk.Button(
            file_frame, text="Классифицировать", 
            command=self.classify_batch, state='disabled'
        )
        self.batch_classify_btn.pack(side='left')
        
        # Прогресс бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.batch_frame, variable=self.progress_var, 
            maximum=100, mode='determinate'
        )
        self.progress_bar.pack(fill='x', pady=(0, 10))
        
        self.progress_label = ttk.Label(self.batch_frame, text="")
        self.progress_label.pack(anchor='w', pady=(0, 10))
        
        # Результаты
        ttk.Label(self.batch_frame, text="Результаты:", 
                  style='Header.TLabel').pack(anchor='w', pady=(0, 5))
        
        result_frame = ttk.Frame(self.batch_frame)
        result_frame.pack(fill='both', expand=True)
        
        columns = ('Индекс', 'Текст', 'Предсказанный класс', 'Вероятность')
        self.batch_tree = ttk.Treeview(result_frame, columns=columns, 
                                       show='headings', height=15)
        
        self.batch_tree.heading('Индекс', text='№')
        self.batch_tree.heading('Текст', text='Текст (первые 50 символов)')
        self.batch_tree.heading('Предсказанный класс', text='Предсказанный класс')
        self.batch_tree.heading('Вероятность', text='Вероятность (%)')
        
        self.batch_tree.column('Индекс', width=50)
        self.batch_tree.column('Текст', width=300)
        self.batch_tree.column('Предсказанный класс', width=150)
        self.batch_tree.column('Вероятность', width=120)
        
        scrollbar = ttk.Scrollbar(result_frame, orient='vertical', 
                                 command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=scrollbar.set)
        
        self.batch_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Кнопка сохранения
        self.save_btn = ttk.Button(self.batch_frame, text="Сохранить результаты", 
                                    command=self.save_results, state='disabled')
        self.save_btn.pack(pady=(10, 0))
        
    def load_model(self):
        """Загрузка модели и словаря"""
        try:
            # Загрузка checkpoint из одного файла
            MODEL_PATH = f"{self.mname_entry.get()}.pth"
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            
            # Создание словаря из checkpoint
            self.vocab = Vocabulary()
            self.vocab.load_from_dict(
                checkpoint['vocab_token2idx'],
                checkpoint['vocab_idx2token']
            )
            
            # Создание модели
            self.model = TextClassifier(
                vocab_size=checkpoint['vocab_size'],
                embedding_dim=checkpoint['embedding_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                num_classes=checkpoint['num_classes'],
                num_layers=checkpoint['num_layers']
            ).to(DEVICE)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Загрузка маппинга меток
            self.label2idx = checkpoint['label2idx']
            self.idx2label = {v: k for k, v in self.label2idx.items()}
            
            self.status_label.config(
                text=f"✓ Модель загружена ({checkpoint['num_classes']} классов, словарь: {checkpoint['vocab_size']} токенов)", 
                foreground='green'
            )
            self.classify_btn.config(state='normal')
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.status_label.config(
                text=f"✗ Ошибка загрузки модели", 
                foreground='red'
            )
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{str(e)}\n\nПодробности:\n{error_details}")
    
    def predict(self, text):
        """Предсказание класса для текста"""
        if self.model is None:
            return None
        
        # Обрезаем текст до максимальной длины
        text = text[:MAX_SEQ_LENGTH]
        
        # Кодируем текст
        encoded = self.vocab.encode(text)
        
        if len(encoded) == 0:
            return None
        
        # Преобразуем в тензор
        text_tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
        length_tensor = torch.tensor([len(encoded)], dtype=torch.long)
        
        # Получаем предсказание
        with torch.no_grad():
            output = self.model(text_tensor, length_tensor)
            probabilities = torch.softmax(output, dim=1)
            
        return probabilities[0].cpu().numpy()
    
    def classify_single(self):
        """Классификация одного текста"""
        text = self.text_input.get('1.0', 'end-1c').strip()
        
        if not text:
            messagebox.showwarning("Предупреждение", "Введите текст для классификации")
            return
        
        # Очищаем предыдущие результаты
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # Получаем предсказание
        probabilities = self.predict(text)
        
        if probabilities is None:
            messagebox.showerror("Ошибка", "Не удалось классифицировать текст")
            return
        
        # Сортируем по вероятности
        top_indices = probabilities.argsort()[::-1][:10]  # Топ-10
        
        # Заполняем таблицу
        for idx in top_indices:
            class_name = self.idx2label[idx]
            prob = probabilities[idx] * 100
            self.result_tree.insert('', 'end', values=(class_name, f'{prob:.2f}'))
    
    def clear_single(self):
        """Очистка полей одиночной классификации"""
        self.text_input.delete('1.0', 'end')
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
    
    def select_file(self):
        """Выбор файла для пакетной классификации"""
        filename = filedialog.askopenfilename(
            title="Выберите CSV файл",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            self.batch_classify_btn.config(state='normal')
    
    def classify_batch(self):
        """Пакетная классификация"""
        filepath = self.file_path_var.get()
        if not filepath:
            return
        
        # Запускаем в отдельном потоке
        thread = threading.Thread(target=self._classify_batch_worker, args=(filepath,))
        thread.daemon = True
        thread.start()
    
    def _classify_batch_worker(self, filepath):
        """Рабочий поток для пакетной классификации"""
        try:
            # Отключаем кнопки
            self.batch_classify_btn.config(state='disabled')
            
            # Загрузка данных
            self.progress_label.config(text="Загрузка данных...")
            df = pd.read_csv(filepath, sep='\t', header=None, encoding='utf-8')
            
            # Определяем формат файла
            if len(df.columns) == 1:
                df.columns = ['text']
                has_labels = False
            else:
                df.columns = ['label', 'text']
                has_labels = True
            
            df['text'] = df['text'].astype(str)
            total = len(df)
            
            # Очищаем таблицу результатов
            for item in self.batch_tree.get_children():
                self.batch_tree.delete(item)
            
            results = []
            
            # Классификация
            for idx, row in df.iterrows():
                text = row['text']
                
                # Предсказание
                probabilities = self.predict(text)
                if probabilities is None:
                    continue
                
                pred_idx = probabilities.argmax()
                pred_class = self.idx2label[pred_idx]
                pred_prob = probabilities[pred_idx] * 100
                
                # Сохраняем результат
                result = {
                    'index': idx,
                    'text': text[:50] + ('...' if len(text) > 50 else ''),
                    'predicted_class': pred_class,
                    'probability': pred_prob
                }
                
                if has_labels:
                    result['true_class'] = row['label']
                
                results.append(result)
                
                # Обновляем прогресс
                progress = (idx + 1) / total * 100
                self.progress_var.set(progress)
                self.progress_label.config(
                    text=f"Обработано: {idx + 1}/{total} ({progress:.1f}%)"
                )
                
                # Добавляем в таблицу (каждые 100 записей)
                if idx % 100 == 0:
                    self.batch_tree.insert('', 'end', values=(
                        idx + 1,
                        result['text'],
                        pred_class,
                        f"{pred_prob:.2f}"
                    ))
            
            # Добавляем оставшиеся результаты
            for result in results[-100:]:
                self.batch_tree.insert('', 'end', values=(
                    result['index'] + 1,
                    result['text'],
                    result['predicted_class'],
                    f"{result['probability']:.2f}"
                ))
            
            # Сохраняем результаты
            self.batch_results = results
            
            # Вычисляем точность если есть метки
            if has_labels:
                correct = sum(1 for r in results if str(r['true_class']) == r['predicted_class'])
                accuracy = correct / len(results) * 100
                self.progress_label.config(
                    text=f"Завершено! Обработано: {total} записей. Точность: {accuracy:.2f}%"
                )
            else:
                self.progress_label.config(text=f"Завершено! Обработано: {total} записей")
            
            # Включаем кнопку сохранения
            self.save_btn.config(state='normal')
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            messagebox.showerror("Ошибка", f"Ошибка при обработке файла:\n{str(e)}\n\n{error_details}")
            self.progress_label.config(text="Ошибка обработки")
        finally:
            self.batch_classify_btn.config(state='normal')
    
    def save_results(self):
        """Сохранение результатов пакетной классификации"""
        if not hasattr(self, 'batch_results'):
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            df = pd.DataFrame(self.batch_results)
            df.to_csv(filename, index=False, encoding='utf-8')
            messagebox.showinfo("Успех", f"Результаты сохранены в {filename}")


def main():
    root = tk.Tk()
    app = TextClassifierApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()