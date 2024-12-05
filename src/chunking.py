import os
import json
import pandas as pd
import re

# Split large text into smaller chunks
def split_text_into_chunks(text, max_length=1000, min_length=200):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence
        else:
            if len(current_chunk.strip()) >= min_length:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# Process all .txt files in the directory and create chunks
def process_books(directory):
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                book_chunks = split_text_into_chunks(text)
                for idx, chunk in enumerate(book_chunks):
                    chunks.append({
                        'book': filename,
                        'chunk_index': idx,
                        'text': chunk,
                        'author': "J.K. Rowling",
                        'source_type': "book"
                    })
    return chunks


# Process all .csv files in the directory and convert rows to dictionaries
def process_csv(directory):
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            try:
                df = pd.read_csv(filepath, sep=',', on_bad_lines='skip', engine='python')
                for _, row in df.iterrows():
                    for column, value in row.items():
                        if isinstance(value, str) and len(value) > 100:
                            text_chunks = split_text_into_chunks(value)
                            for idx, chunk in enumerate(text_chunks):
                                chunks.append({
                                    'source': filename,
                                    'chunk_index': idx,
                                    'text': chunk,
                                    'column': column
                                })
            except Exception as e:
                print(f"Processing error {filename}: {e}")
    return chunks

# Save chunks into a JSON file
def save_chunks(chunks, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

# Merge multiple JSON files containing chunks into one
def merge_chunks(files, output_file="merged_chunks.json"):
    merged_chunks = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            for chunk in chunks:
                if 'text' in chunk and chunk['text'].strip():
                    merged_chunks.append(chunk)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_chunks, f, ensure_ascii=False, indent=2)