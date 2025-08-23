import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Экстракторы ключевых слов

def extract_keywords_yake(text: str, *, max_keywords: int = 10) -> list[str]:
    """Извлечь ключевые слова из текста с помощью YAKE."""
    import yake  # type: ignore

    extractor = yake.KeywordExtractor(lan="en", n=1, top=max_keywords)
    scored_keywords = extractor.extract_keywords(text)
    return [kw for kw, _ in scored_keywords]

_NLP = None
def extract_keywords_spacy(text: str, *, max_keywords: int = 10) -> list[str]:
    global _NLP
    try:
        import spacy  # type: ignore
    except ImportError as e:
        raise ImportError("Для extract_keywords_spacy требуется установленный пакет spaCy.") from e

    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    doc = _NLP(text.lower())
    keywords: list[str] = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and token.is_alpha and not token.is_stop:
            keywords.append(token.lemma_)
            if len(keywords) >= max_keywords:
                break
    return keywords


def extract_keywords_textrank(text: str, *, max_keywords: int = 10) -> list[str]:
    from summa import keywords as summa_keywords  # type: ignore

    kw_string: str = summa_keywords.keywords(text, words=max_keywords)
    return [kw.strip() for kw in kw_string.split("\n") if kw.strip()]


def extract_keywords_keybert(text: str, *, max_keywords: int = 10, model=None) -> list[str]:
    from keybert import KeyBERT  # type: ignore

    kw_model = model if model is not None else KeyBERT()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=max_keywords,
    )
    return [kw for kw, _ in keywords]


def extract_keywords_tfidf(corpus: pd.Series, *, max_keywords: int = 10) -> list[list[str]]:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    tfidf_keywords: list[list[str]] = []
    for row in tfidf_matrix:
        row_array = row.toarray().flatten()
        sorted_indices = row_array.argsort()[::-1]
        keywords = [feature_names[idx] for idx in sorted_indices if row_array[idx] > 0][:max_keywords]
        tfidf_keywords.append(keywords)
    return tfidf_keywords

# Пакетная обработка

def process_batch(batch_df: pd.DataFrame, methods: list[str], *, keybert_model=None) -> dict[str, list[list[str]]]:
    results: dict[str, list[list[str]]] = {method: [] for method in methods}

    for _, row in batch_df.iterrows():
        text: str = row["Abstract"]

        if "YAKE" in methods:
            results["YAKE"].append(extract_keywords_yake(text))
        if "spaCy" in methods:
            results["spaCy"].append(extract_keywords_spacy(text))
        if "TextRank" in methods:
            results["TextRank"].append(extract_keywords_textrank(text))
        if "KeyBERT" in methods:
            results["KeyBERT"].append(extract_keywords_keybert(text, model=keybert_model))

    return results

def process_data_in_batches(
    df: pd.DataFrame,
    *,
    batch_size: int = 1000,
    methods: list[str] | None = None,
    keybert_model=None,
) -> dict[str, list[list[str]]]:
    if methods is None:
        methods = ["YAKE", "spaCy", "TextRank", "KeyBERT"]

    results: dict[str, list[list[str]]] = {method: [] for method in methods}
    num_batches = (len(df) + batch_size - 1) // batch_size

    print(f"Обработка {len(df)} строк в {num_batches} батчах...")
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_df = df.iloc[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_results = process_batch(batch_df, methods, keybert_model=keybert_model)
        for method in methods:
            results[method].extend(batch_results[method])

    return results

# Оркестратор

def run_keyword_extraction(
    df: pd.DataFrame,
    *,
    methods: list[str] | None = None,
    batch_size: int = 1000,
    tfidf_top: int = 10,
    keybert_model=None,
) -> pd.DataFrame:
    if methods is None:
        methods = ["YAKE", "spaCy", "TextRank", "KeyBERT"]

    if "KeyBERT" in methods and keybert_model is None:
        from keybert import KeyBERT  # импортируем только при необходимости
        keybert_model = KeyBERT()

    extraction_results = process_data_in_batches(
        df,
        batch_size=batch_size,
        methods=methods,
        keybert_model=keybert_model,
    )

    for method in methods:
        df[f"{method}_Keywords"] = extraction_results[method]

    df["TFIDF_Keywords"] = extract_keywords_tfidf(df["Abstract"], max_keywords=tfidf_top)

    return df


if __name__ == "__main__":
    try:
        # cleaned_df = pd.read_csv("cleaned_data.csv")
        # results_df = run_keyword_extraction(cleaned_df, batch_size=500)
        # print(results_df.head())
        pass
    except Exception as e:
        print(f"Error during demonstration: {e}")
