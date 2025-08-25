import spacy
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Загружаем модель spaCy для английского
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
tqdm.pandas()

# Гибридная нормализация ключевых слов

def normalize_keywords_hybrid(keywords):
    if not keywords or not isinstance(keywords, list):
        return []

    parsed_phrases = []
    for phrase in keywords:
        doc = nlp(phrase.lower())
        tokens = [
            token.lemma_
            for token in doc
            if token.pos_ in {"NOUN", "ADJ", "PROPN"} and token.is_alpha and not token.is_stop
        ]
        if tokens:
            parsed_phrases.append((" ".join(tokens), set(tokens)))

    # Сохраняем многословные токены
    multiword_tokens = {
        token for _, tokens in parsed_phrases if len(tokens) > 1 for token in tokens
    }

    final_phrases = []
    for text, tokens in parsed_phrases:
        # Убираем однословные фразы, если они уже входят в состав многословных
        if len(tokens) == 1 and next(iter(tokens)) in multiword_tokens:
            continue
        final_phrases.append(text)

    return sorted(set(final_phrases))

# Применение нормализации

df["Extracted Keywords"] = df["Extracted Keywords"].apply(
    lambda x: x if isinstance(x, list) else []
)
df["Normalized Keywords"] = df["Extracted Keywords"].progress_apply(normalize_keywords_hybrid)

# Метрики: количество ключевых слов и уникальные слова

def avg_keyword_count(df, col):
    """Среднее количество ключевых слов."""
    return df[col].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()

def unique_word_count(df, col):
    """Количество уникальных ключевых слов."""
    all_words = [word for sublist in df[col] if isinstance(sublist, list) for word in sublist]
    return len(set(all_words))

# Считаем пересечения и семантическую близость

model = SentenceTransformer("all-MiniLM-L6-v2")

def list_to_string(lst):
    return " ".join(lst) if isinstance(lst, list) else ""

def keyword_overlap(row):
    """Доля пересечения оригинальных и нормализованных ключевых слов."""
    if isinstance(row["Extracted Keywords"], list) and isinstance(row["Normalized Keywords"], list):
        orig_set = set(row["Extracted Keywords"])
        norm_set = set(row["Normalized Keywords"])
        return len(orig_set & norm_set) / len(orig_set) if len(orig_set) > 0 else 0
    return 0

def semantic_similarity(row):
    """Семантическая близость между наборами ключевых слов."""
    if isinstance(row["Extracted Keywords"], list) and isinstance(row["Normalized Keywords"], list):
        orig_text = list_to_string(row["Extracted Keywords"])
        norm_text = list_to_string(row["Normalized Keywords"])
        try:
            emb_orig = model.encode([orig_text])[0]
            emb_norm = model.encode([norm_text])[0]
            return cosine_similarity([emb_orig], [emb_norm])[0][0]
        except:
            return np.nan
    return np.nan

# Вычисляем метрики

metrics = {}
metrics["avg_len_original"] = avg_keyword_count(df, "Extracted Keywords")
metrics["avg_len_normalized"] = avg_keyword_count(df, "Normalized Keywords")
metrics["unique_words_original"] = unique_word_count(df, "Extracted Keywords")
metrics["unique_words_normalized"] = unique_word_count(df, "Normalized Keywords")

df["overlap_ratio"] = df.apply(keyword_overlap, axis=1)
metrics["mean_overlap_ratio"] = df["overlap_ratio"].mean()

df["semantic_sim"] = df.apply(semantic_similarity, axis=1)
metrics["mean_semantic_similarity"] = df["semantic_sim"].mean()

# Удаление дублирующих слов внутри фраз

def remove_duplicate_word_phrases(keyword_list):
    clean_phrases = []
    for phrase in keyword_list:
        words = phrase.split()
        if len(words) == len(set(words)):  # оставляем фразы без дубликатов
            clean_phrases.append(phrase)
    return clean_phrases

df["Normalized Keywords"] = df["Normalized Keywords"].apply(remove_duplicate_word_phrases)

# Сохраняем результат

df.to_csv("Big_data.csv", index=False)

print("\n=== Метрики по ключевым словам ===")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")
