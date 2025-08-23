import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Построение топ-ключевых слов для каждой модели

def get_top_keywords(results_dict, models, top_n=10):
    top_keywords = {}
    for model in models:
        # Собираем все ключевые слова в один список
        all_keywords = [kw for kws in results_dict[model] for kw in kws]
        counter = Counter(all_keywords)
        top_keywords[model] = counter.most_common(top_n)
    return top_keywords

# Получаем топ-10 ключевых слов
models = ['YAKE', 'spaCy', 'TextRank', 'KeyBERT', 'TF-IDF']
top_keywords = get_top_keywords(results, models, top_n=10)

# Визуализация топ-ключевых слов для каждой модели
def plot_top_keywords(top_keywords):
    num_models = len(top_keywords)
    fig, axes = plt.subplots(num_models, 1, figsize=(10, 6 * num_models), squeeze=False)

    for i, (model, keywords) in enumerate(top_keywords.items()):
        words = [kw[0] for kw in keywords]
        counts = [kw[1] for kw in keywords]

        axes[i, 0].barh(words, counts, color='skyblue')
        axes[i, 0].set_title(f"Топ-10 ключевых слов — {model}", fontsize=14)
        axes[i, 0].invert_yaxis()
        axes[i, 0].grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

plot_top_keywords(top_keywords)

# Среднее количество ключевых слов

def compare_average_keyword_count(results_dict, models):
    """
    Сравнивает среднее количество ключевых слов для каждой модели.
    """
    avg_counts = {}
    for model in models:
        total_keywords = sum(len(kws) for kws in results_dict[model])
        avg_counts[model] = total_keywords / len(results_dict[model])
    return avg_counts

avg_keyword_counts = compare_average_keyword_count(results, models)

plt.figure(figsize=(8, 5))
plt.bar(avg_keyword_counts.keys(), avg_keyword_counts.values(), color='lightgreen')
plt.title("Среднее количество ключевых слов по моделям")
plt.ylabel("Среднее количество")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Анализ пересечений ключевых слов между моделями

def compute_intersections(results_dict, models):
    """
    Считает пересечение ключевых слов между всеми моделями.
    """
    sets_dict = {m: set(kw for kws in results_dict[m] for kw in kws) for m in models}
    intersections = {}

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            inter = sets_dict[m1] & sets_dict[m2]
            intersections[(m1, m2)] = len(inter)

    return intersections

intersections = compute_intersections(results, models)

print("\nПересечение ключевых слов между моделями:")
for pair, count in intersections.items():
    print(f"{pair[0]} & {pair[1]} → {count}")

# 3.4. Подсчёт уникальных ключевых слов

def get_unique_keyword_counts(results_dict, models):
    """
    Считает количество уникальных ключевых слов для каждой модели.
    """
    unique_counts = {}
    for model in models:
        keywords = [kw for kws in results_dict[model] for kw in kws]
        unique_counts[model] = len(set(keywords))
    return unique_counts

unique_keyword_counts = get_unique_keyword_counts(results, models)

plt.figure(figsize=(8, 5))
plt.bar(unique_keyword_counts.keys(), unique_keyword_counts.values(), color='coral')
plt.title("Количество уникальных ключевых слов")
plt.ylabel("Число уникальных слов")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Семантическая схожесть ключевых слов

model_st = SentenceTransformer("all-MiniLM-L6-v2")

def calc_semantic_similarity(df):
    """
    Считает косинусную схожесть между исходными и нормализованными ключевыми словами.
    """
    def join_words(lst):
        return " ".join(lst) if isinstance(lst, list) else ""

    df["Text_Original"] = df["Extracted Keywords"].apply(join_words)
    df["Text_Normalized"] = df["Normalized Keywords"].apply(join_words)

    emb_orig = model_st.encode(df["Text_Original"].tolist(), convert_to_tensor=True)
    emb_norm = model_st.encode(df["Text_Normalized"].tolist(), convert_to_tensor=True)

    similarities = cosine_similarity(emb_orig.cpu(), emb_norm.cpu())
    df["Semantic Similarity"] = np.diag(similarities)
    return df

df = calc_semantic_similarity(df)

print("\nСредняя семантическая схожесть:", df["Semantic Similarity"].mean())

# Поиск вложенных и дублирующихся ключевых слов

def find_nested_keywords(df):
    """
    Находит ключевые слова, которые являются частью других словосочетаний.
    """
    nested_cases = []
    for idx, keywords in enumerate(df["Normalized Keywords"]):
        nested_pairs = []
        for i, kw1 in enumerate(keywords):
            for j, kw2 in enumerate(keywords):
                if i != j and kw1 in kw2.split():
                    nested_pairs.append((kw1, kw2))
        if nested_pairs:
            nested_cases.append((idx, nested_pairs))
    return nested_cases

nested_cases = find_nested_keywords(df)
print(f"Найдено строк с вложенными словами: {len(nested_cases)}")

def remove_duplicate_word_phrases(keyword_list):
    """
    Удаляет словосочетания с дублирующимися словами.
    """
    clean_phrases = []
    for phrase in keyword_list:
        words = phrase.split()
        if len(words) == len(set(words)):
            clean_phrases.append(phrase)
    return clean_phrases

df["Normalized Keywords"] = df["Normalized Keywords"].apply(remove_duplicate_word_phrases)
