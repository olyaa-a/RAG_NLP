# 📚 RAG для Гаррі Поттера

**Система Retrieval‑Augmented Generation**, що відповідає на запити користувачів, використовуючи книги та фільми про Гаррі Поттера як джерела знань.

---

## 🎯 Задача

Розробити систему RAG, яка:
- Індексуватиме тексти книг/фільмів про Гаррі Поттера
- Відповідатиме на запити з точними цитатами й джерелами

---

## 🔧 Архітектура

### Data source
- Використано датасети з Kaggle:
  - *Harry Potter Dataset*
  - *Harry Potter Books*
- Джерела (.txt, .csv) зберігаються локально

### Chunking
- Розбиває текст на чанки (200–1000 символів)
  - Для `.txt`: додається назва книги, автор, номер чанка
  - Для `.csv`: ділимо значення колонок на чанки
- Зберігає всі чанки у форматі `chunks.json`

### Retriever
1. **BM25** – ключове слово
2. **Dense** – семантичний пошук з `all-MiniLM-L6-v2`
3. **Hybrid** – комбінований метод обох

**Приклади запитів:**
- BM25:
  - "What type of wood is used for Harry Potter's wand?"
  - `"- After all this time? – Always." What is that?`
- Dense:
  - "Why do wands choose their owners?"
  - "Explain why Severus Snape was both a hero and a villain."

### LLM
- Модель: `groq/llama3-8b-8192` через Groq Cloud API
- Обробляє перші 5 чанків (до 4096 символів)
- Генерує відповідь, цитати та опис джерел

### Reranker
- **N/A**

### Citations
- Відповіді супроводжуються цитатами в квадратних дужках
- Якщо LLM доступна — генерується детальний опис джерел, інакше — простий список

### UI (Gradio)
- Поля:
  - API key
  - Поле запиту
  - Вибір методу пошуку (BM25 / Dense / Hybrid)
  - Кнопка **Search**
  - Три області відображення:
    1. Відповідь
    2. Результати з контекстом
    3. Список джерел

---

## 🚀 Демо
Спробуй онлайн: **[Hugging Face Space](посилання на запущений сервіс)**

---

## 🧰 Встановлення та запуск

```bash
git clone https://github.com/olyaa-a/RAG_NLP.git
cd RAG_NLP

pip install -r requirements.txt

# Експорт API_KEY
export API_KEY=<твій API ключ>

python app.py
