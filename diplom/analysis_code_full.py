"""
Analysis code extracted from the Master's thesis (ВКР)

This script contains key functions that were described in the final
pages of the thesis. The code demonstrates how the project cleans
textual data, processes it in batches, extracts keywords using
multiple models (YAKE, spaCy, TextRank and KeyBERT), computes
TF‑IDF‑based keywords and builds graphs of relationships between
keywords and authors.  Some functions are left as placeholders
(`extract_keywords_yake`, `extract_keywords_spacy`, etc.) since
their implementations were not provided in the thesis.  You can
replace these stubs with your own implementations.

Requirements:
    pandas, tqdm, sklearn, networkx, numpy (for combinations)

The script is for demonstration purposes and may require
adjustments to run on your own data.
"""

import re
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx


def clean_abstract_column(df: pd.DataFrame, column_name: str = 'Abstract') -> pd.DataFrame:
    """Clean the Abstract column by removing duplicates, missing values,
    stripping whitespace, removing HTML tags and splitting camelCase.

    Args:
        df: DataFrame with at least the column_name field.
        column_name: The name of the text column to clean.

    Returns:
        A cleaned DataFrame with the specified text column processed.
    """
    # Drop duplicates and rows with missing abstracts
    df = df.drop_duplicates(subset=[column_name]).dropna(subset=[column_name])
    # Remove entries shorter than 10 characters
    df = df[df[column_name].str.strip().str.len() > 10]
    # Strip any HTML tags
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'<.*?>', '', x))
    # Remove JATS and HTML encoded tags
    df = df[~df[column_name].str.contains(r'<jats:p>|&lt;', regex=True)]
    # Insert a space between camelCase words
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', x))
    return df


def process_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the dataset from a CSV file.

    The dataset is expected to contain the following columns: Author/s,
    Abstract, Source Country and Keywords.  Unwanted rows are dropped and
    the abstract text is cleaned.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Cleaned DataFrame containing only the key columns.
    """
    df = pd.read_csv(file_path)
    key_columns = ['Author/s', 'Abstract', 'Source Country', 'Keywords']
    df = df[key_columns]
    df.dropna(subset=['Author/s', 'Source Country', 'Keywords'], inplace=True)
    df = clean_abstract_column(df, column_name='Abstract')
    return df

# ---------------------------------------------------------------------------
# Keyword Extraction and Analysis Helpers
# ---------------------------------------------------------------------------


def get_top_keywords(
    poprobovat: Dict[str, List[List[str]]],
    models: List[str],
    top_n: int = 10
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Aggregate keyword lists across a set of models and return the top N keywords
    by frequency for each model.

    Args:
        poprobovat: mapping from model name to list of keyword lists. Each
            keyword list corresponds to one publication.
        models: list of model names to consider (e.g. ['YAKE','spaCy','TextRank','KeyBERT']).
        top_n: number of top keywords to return.

    Returns:
        Dictionary mapping model name to a list of (keyword, frequency) tuples.
    """
    top_keywords: Dict[str, List[Tuple[str, int]]] = {}
    for model in models:
        counter: Counter = Counter()
        for kw_list in poprobovat.get(model, []):
            counter.update(kw_list)
        top_keywords[model] = counter.most_common(top_n)
    return top_keywords


def compare_average_keyword_count(poprobovat: Dict[str, List[List[str]]], models: List[str]) -> Dict[str, float]:
    """
    Compute the average number of keywords per publication for each model.

    Args:
        poprobovat: mapping from model to lists of keywords per publication.
        models: list of models to consider.

    Returns:
        Dictionary mapping model name to average keyword count.
    """
    averages: Dict[str, float] = {}
    for model in models:
        lengths = [len(lst) for lst in poprobovat.get(model, [])]
        averages[model] = sum(lengths) / len(lengths) if lengths else 0.0
    return averages


def normalize_keywords_hybrid(keywords: List[str]) -> List[str]:
    """
    Simple hybrid keyword normalization.

    This function lowercases all keywords, strips surrounding whitespace
    and removes duplicates while preserving order. In the thesis a more
    sophisticated approach was implemented (lemmatization, stemming). This
    simplified version demonstrates the overall pipeline but can be
    replaced with a full implementation when available.

    Args:
        keywords: list of raw keyword strings.

    Returns:
        List of normalized keywords without duplicates.
    """
    seen = set()
    normalized: List[str] = []
    for kw in keywords:
        kw_norm = kw.lower().strip()
        if kw_norm and kw_norm not in seen:
            seen.add(kw_norm)
            normalized.append(kw_norm)
    return normalized


def list_to_string(lst: List[str]) -> str:
    """
    Convert a list of strings to a comma-separated string.

    Args:
        lst: list of strings.

    Returns:
        A single string with elements separated by ', '.
    """
    return ', '.join(lst)


def avg_keyword_count(df: pd.DataFrame, col: str) -> float:
    """
    Calculate the average number of keywords in the specified column.

    Args:
        df: DataFrame containing a list-like column.
        col: name of the column with keyword lists.

    Returns:
        Average number of keywords per row.
    """
    counts = df[col].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)
    return counts.mean()


def unique_word_count(df: pd.DataFrame, col: str) -> int:
    """
    Count the total number of unique keywords across all lists in a column.

    Args:
        df: DataFrame with a list-like column.
        col: name of the column with keyword lists.

    Returns:
        Number of distinct keywords across all rows.
    """
    unique_words = set()
    for lst in df[col]:
        if isinstance(lst, list):
            unique_words.update(lst)
    return len(unique_words)


def keyword_overlap(row: pd.Series) -> float:
    """
    Compute the ratio of overlap between two keyword lists stored in a row.

    This helper expects the row to contain two list-like columns named
    'Keywords_A' and 'Keywords_B'. It returns the Jaccard similarity
    between them (intersection over union). This metric was used in
    the thesis to compare results from different extraction models.

    Args:
        row: row with 'Keywords_A' and 'Keywords_B' lists.

    Returns:
        Jaccard similarity as a float between 0 and 1.
    """
    a = set(row.get('Keywords_A') or [])
    b = set(row.get('Keywords_B') or [])
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def semantic_similarity(row: pd.Series, model: 'SentenceTransformer') -> float:
    """
    Compute semantic similarity between two lists of keywords using
    sentence embeddings.

    Args:
        row: row with 'Keywords_A' and 'Keywords_B' lists.
        model: a sentence transformer model for generating embeddings.

    Returns:
        Cosine similarity between the averaged embeddings of both lists.
    """
    a = row.get('Keywords_A') or []
    b = row.get('Keywords_B') or []
    if not a or not b:
        return 0.0
    emb_a = model.encode(a, convert_to_tensor=True).mean(axis=0)
    emb_b = model.encode(b, convert_to_tensor=True).mean(axis=0)
    sim = float(util.pytorch_cos_sim(emb_a, emb_b))
    return sim


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate evaluation metrics to compare keyword extraction models.

    The thesis considered metrics such as the average keyword count,
    unique keyword count, keyword overlap and semantic similarity. This
    function demonstrates how to compute these metrics across pairs of
    models stored in the DataFrame. It assumes that the DataFrame has
    columns named like 'ModelA Keywords' and 'ModelB Keywords' for each
    pair and constructs a new DataFrame with the calculated metrics.

    Args:
        df: DataFrame with paired keyword columns (e.g. 'TF-IDF Keywords',
            'YAKE Keywords', etc.).

    Returns:
        DataFrame summarizing evaluation metrics for each pair of models.
    """
    # Identify keyword columns by suffix 'Keywords'
    keyword_cols = [col for col in df.columns if 'Keywords' in col]
    results = []
    # Preload sentence transformer only if needed
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Compare each pair of keyword columns
    for i, col_a in enumerate(keyword_cols):
        for col_b in keyword_cols[i+1:]:
            tmp = df[[col_a, col_b]].rename(columns={col_a: 'Keywords_A', col_b: 'Keywords_B'})
            avg_a = avg_keyword_count(df, col_a)
            avg_b = avg_keyword_count(df, col_b)
            overlap_scores = tmp.apply(keyword_overlap, axis=1)
            sem_scores = tmp.apply(lambda r: semantic_similarity(r, model), axis=1)
            results.append({
                'Model A': col_a,
                'Model B': col_b,
                'Average Keywords A': avg_a,
                'Average Keywords B': avg_b,
                'Average Overlap': overlap_scores.mean(),
                'Average Semantic Similarity': sem_scores.mean(),
            })
    return pd.DataFrame(results)


def extract_keywords_yake(text: str) -> List[str]:
    """Placeholder for YAKE keyword extraction.

    Replace this stub with an actual implementation from the YAKE library.
    """
    raise NotImplementedError("Replace with YAKE keyword extraction implementation")


def extract_keywords_spacy(text: str) -> List[str]:
    """Placeholder for spaCy keyword extraction.

    Replace this stub with an actual implementation using spaCy NLP models.
    """
    raise NotImplementedError("Replace with spaCy keyword extraction implementation")


def extract_keywords_textrank(text: str) -> List[str]:
    """Placeholder for TextRank keyword extraction.

    Replace this stub with an actual implementation of the TextRank algorithm.
    """
    raise NotImplementedError("Replace with TextRank keyword extraction implementation")


def extract_keywords_keybert(text: str) -> List[str]:
    """Placeholder for KeyBERT keyword extraction.

    Replace this stub with an actual implementation using KeyBERT.
    """
    raise NotImplementedError("Replace with KeyBERT keyword extraction implementation")


def process_batch(batch_df: pd.DataFrame, methods: List[str]) -> Dict[str, List[List[str]]]:
    """Process a batch of rows and extract keywords using the specified methods.

    Args:
        batch_df: Subset of the DataFrame to process.
        methods: List of methods to apply. Supported values: 'YAKE', 'spaCy',
            'TextRank', 'KeyBERT'.

    Returns:
        Dictionary mapping each method to a list of keyword lists.
    """
    results = {method: [] for method in methods}
    for _, row in batch_df.iterrows():
        text = row['Abstract']
        if 'YAKE' in methods:
            results['YAKE'].append(extract_keywords_yake(text))
        if 'spaCy' in methods:
            results['spaCy'].append(extract_keywords_spacy(text))
        if 'TextRank' in methods:
            results['TextRank'].append(extract_keywords_textrank(text))
        if 'KeyBERT' in methods:
            results['KeyBERT'].append(extract_keywords_keybert(text))
    return results


def process_data_in_batches(df: pd.DataFrame, batch_size: int = 1000,
                            methods: List[str] = None) -> Dict[str, List[List[str]]]:
    """Process the entire DataFrame in batches.

    Divides the data into chunks to avoid memory issues when working with
    large datasets. For each batch it calls `process_batch` and
    aggregates the results.

    Args:
        df: DataFrame to process.
        batch_size: Number of rows per batch.
        methods: Keyword extraction methods to apply.

    Returns:
        Dictionary mapping each method to a list of keyword lists for
        the entire dataset.
    """
    if methods is None:
        methods = ['YAKE', 'spaCy', 'TextRank', 'KeyBERT']
    results = {method: [] for method in methods}
    num_batches = (len(df) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Processing batches"):
        batch_df = df.iloc[i * batch_size:(i + 1) * batch_size]
        batch_results = process_batch(batch_df, methods)
        for method in methods:
            results[method].extend(batch_results[method])
    return results


def compute_tfidf_keywords(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Compute TF-IDF keywords for each abstract and append as a new column.

    Args:
        df: DataFrame containing an 'Abstract' column.
        top_n: Number of top keywords to extract for each row.

    Returns:
        DataFrame with an additional 'TF-IDF Keywords' column containing a
        list of keywords for each abstract.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Abstract'])
    feature_names = vectorizer.get_feature_names_out()
    keywords_list = []
    for row in tfidf_matrix:
        row_array = row.toarray().flatten()
        sorted_indices = row_array.argsort()[::-1]
        keywords = [feature_names[i] for i in sorted_indices[:top_n]]
        keywords_list.append(keywords)
    df = df.copy()
    df['TF-IDF Keywords'] = keywords_list
    return df


def build_keyword_cooccurrence_graph(df: pd.DataFrame, top_keywords_limit: int = 500) -> nx.Graph:
    """Build an undirected co-occurrence graph for keywords.

    Nodes represent keywords and edges connect keywords that appear in the
    same publication. Only keywords in the top `top_keywords_limit` by
    frequency are considered.

    Args:
        df: DataFrame with a 'Normalized Keywords' column containing lists of
            keywords per row.
        top_keywords_limit: Number of top keywords to include.

    Returns:
        A NetworkX Graph object representing keyword co-occurrences.
    """
    all_keywords = [kw for keywords in df['Normalized Keywords'] for kw in keywords]
    keyword_counts = Counter(all_keywords)
    top_keywords = set([kw for kw, _ in keyword_counts.most_common(top_keywords_limit)])
    G = nx.Graph()
    for keywords in df['Normalized Keywords']:
        filtered = [kw for kw in keywords if kw in top_keywords]
        for k1, k2 in combinations(set(filtered), 2):
            if G.has_edge(k1, k2):
                G[k1][k2]['weight'] += 1
            else:
                G.add_edge(k1, k2, weight=1)
    return G


def build_bipartite_author_keyword_graph(df: pd.DataFrame, top_keywords_limit: int = 500) -> nx.Graph:
    """Build a bipartite graph linking authors and keywords.

    Nodes on one side are authors, on the other side are keywords. An edge
    exists between an author and a keyword if the author has at least
    one publication with that keyword.

    Args:
        df: DataFrame with 'Author/s' and 'Normalized Keywords' columns.
        top_keywords_limit: Number of top keywords to include.

    Returns:
        A NetworkX Graph object representing the bipartite author-keyword
        relationships.
    """
    all_keywords = [kw for keywords in df['Normalized Keywords'] for kw in keywords]
    keyword_counts = Counter(all_keywords)
    top_keywords = set([kw for kw, _ in keyword_counts.most_common(top_keywords_limit)])
    B = nx.Graph()
    for _, row in df.iterrows():
        authors_raw = row['Author/s']
        # Split authors by semicolon and clean extra spaces
        authors = [a.strip() for a in str(authors_raw).split(';') if a.strip()]
        keywords = [kw for kw in row['Normalized Keywords'] if kw in top_keywords]
        for author in authors:
            B.add_node(author, bipartite=0)
            for kw in keywords:
                B.add_node(kw, bipartite=1)
                B.add_edge(author, kw)
    return B

# The following placeholder imports are used by safe_eval and community lookup functions.
import ast


def safe_eval(val):
    """Safely evaluate a string representation of a Python literal.

    Used to convert string representations of lists back into Python lists.
    """
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except Exception:
        return []


def build_community_lookup(df: pd.DataFrame) -> Dict[int, str]:
    """Create a lookup for community IDs to human-readable names.

    Assumes df has a 'Keyword_Communities' column with lists of (word,
    community_id) pairs. Returns a dictionary mapping community IDs to
    comma‑separated strings of the most common words in that community.
    """
    comm_to_words = defaultdict(list)
    for row in df['Keyword_Communities']:
        for word, comm_id in row:
            comm_to_words[comm_id].append(word)
    comm_names = {
        comm_id: ", ".join([kw for kw, _ in Counter(words).most_common(3)])
        for comm_id, words in comm_to_words.items()
    }
    return comm_names


if __name__ == "__main__":
    # Example usage (requires actual data):
    # df = process_data('data.csv')
    # methods = ['YAKE', 'spaCy', 'TextRank', 'KeyBERT']
    # results = process_data_in_batches(df, batch_size=1000, methods=methods)
    # df = compute_tfidf_keywords(df)
    # print(df.head())
    # keyword_graph = build_keyword_cooccurrence_graph(df)
    # author_keyword_graph = build_bipartite_author_keyword_graph(df)
    pass
