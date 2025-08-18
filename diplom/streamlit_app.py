"""
Streamlit Interface for Research Collaboration Search
--------------------------------------------------

This web application provides an interactive interface to search for research
topics and find potential collaboration opportunities based on keywords
extracted from academic abstracts.  It was reconstructed from the thesis
"Методы и программные решения выявления и анализа сетей научного
сотрудничества" and re‑implements the core functionality described in the
final chapters.  The application allows users to:

* Enter a keyword and browse topic communities derived from keyword
  clustering (using Louvain communities).
* Select one or more keywords within a chosen topic to narrow the focus.
* Discover other communities that share publications with the selected
  keywords and explore related keywords for those communities.
* Select a second community and keywords to identify authors publishing
  across both topics.

The underlying data must be exported from the Big Data CSV prepared
during the thesis work.  See ``analysis_code.py`` for the data
cleaning, keyword extraction and community detection pipeline.

Note: this script is meant to run with ``streamlit run streamlit_app.py``.
"""

import ast
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple

import pandas as pd
import streamlit as st



def safe_eval(val):
    """Safely evaluate a string representation of a Python literal.

    The thesis stored lists of keywords and community assignments as
    strings in the CSV; this helper converts those strings back into
    Python objects.  Invalid values return an empty list.

    Args:
        val: a cell value from a pandas DataFrame.

    Returns:
        A Python object or the original value if it is not a string.
    """
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_data(path: str = "Big_data.csv") -> Tuple[pd.DataFrame, dict]:
    """Load the dataset and construct a community name lookup.

    The CSV is expected to include the following columns (see
    ``analysis_code.py``): ``Author/s`` (list of authors separated by
    semicolons), ``Normalized Keywords`` (stringified list of keywords),
    and ``Keyword Communities`` (stringified list of (keyword, community
    id) pairs).  Missing authors are replaced with 'Неизвестный'.  The
    community lookup maps each community id to a comma‑separated string
    of its three most frequent keywords.

    Args:
        path: location of the CSV file relative to this script.

    Returns:
        A tuple ``(df, comm_names)`` where ``df`` is the cleaned
        DataFrame and ``comm_names`` is a mapping of community id to
        descriptive name.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Author/s": "Author_s"})
    df["Author_s"] = df["Author_s"].fillna("Неизвестный")
    df = df.rename(columns={"Keyword Communities": "Keyword_Communities"})
    # Convert string columns back to Python lists
    df["Normalized Keywords"] = df["Normalized Keywords"].apply(safe_eval)
    df["Keyword_Communities"] = df["Keyword_Communities"].apply(safe_eval)
    # Build community name lookup using the three most common keywords per community
    comm_to_words: defaultdict[int, List[str]] = defaultdict(list)
    for row in df["Keyword_Communities"]:
        for word, comm_id in row:
            comm_to_words[comm_id].append(word)
    comm_names = {
        comm_id: ", ".join([kw for kw, _ in Counter(words).most_common(3)])
        for comm_id, words in comm_to_words.items()
    }
    return df, comm_names


def get_keywords_by_comm(df: pd.DataFrame, comm_id: int) -> List[str]:
    """Return sorted unique keywords associated with the given community id."""
    keywords = set()
    for row in df["Keyword_Communities"]:
        for kw, cid in row:
            if cid == comm_id:
                keywords.add(kw)
    return sorted(list(keywords))


def find_suggested_communities(
    df: pd.DataFrame, selected_comm_id: int, selected_keywords: Iterable[str]
) -> Tuple[List[int], dict]:
    """Find communities that co‑occur with the selected keywords.

    Iterates over the DataFrame rows and, for each publication that
    contains all of the ``selected_keywords`` in ``selected_comm_id``,
    collects other communities and their associated keywords.  The result
    is a list of candidate community ids and a mapping of each id to the
    set of related keywords.

    Args:
        df: the DataFrame loaded by :func:`load_data`.
        selected_comm_id: the id of the currently selected community.
        selected_keywords: keywords chosen within the current community.

    Returns:
        A tuple ``(comm_ids, related_keywords)`` where ``comm_ids`` is
        the sorted list of suggested community ids and
        ``related_keywords`` maps each id to its related keywords.
    """
    selected_kw_set = set((kw, selected_comm_id) for kw in selected_keywords)
    suggested_comm_ids = set()
    related_keywords: defaultdict[int, set] = defaultdict(set)
    for _, row in df.iterrows():
        kw_comm_set = set(row["Keyword_Communities"])
        if selected_kw_set.issubset(kw_comm_set):
            for other_kw, other_comm_id in row["Keyword_Communities"]:
                if other_comm_id != selected_comm_id:
                    suggested_comm_ids.add(other_comm_id)
                    related_keywords[other_comm_id].add(other_kw)
    return sorted(list(suggested_comm_ids)), related_keywords


def find_recommended_authors(
    df: pd.DataFrame,
    first_comm_id: int,
    first_keywords: Iterable[str],
    second_comm_id: int,
    second_keywords: Iterable[str],
) -> List[Tuple[str, List[str]]]:
    """Identify authors publishing in both selected topics.

    For each row, check that it contains all (keyword, comm_id) pairs
    from both the first and second selection.  Return a list of
    (author, keyword list) tuples for matching authors.

    Args:
        df: DataFrame of publications.
        first_comm_id: id of the primary community.
        first_keywords: keywords chosen in the primary community.
        second_comm_id: id of the secondary community.
        second_keywords: keywords chosen in the secondary community.

    Returns:
        A list of tuples ``(author_name, normalized_keywords)`` for
        authors who meet the criteria.
    """
    first_set = set((kw, first_comm_id) for kw in first_keywords)
    second_set = set((kw, second_comm_id) for kw in second_keywords)
    recommended = []
    for _, row in df.iterrows():
        kw_comm_set = set(row["Keyword_Communities"])
        if first_set.issubset(kw_comm_set) and second_set.issubset(kw_comm_set):
            recommended.append((row["Author_s"], row["Normalized Keywords"]))
    return recommended


def main() -> None:
    df, comm_names = load_data()
    st.title("Поиск научного сотрудничества по ключевым словам")

    # --- Topic selection ---
    query = st.text_input("Введите ключевое слово для поиска тематики")
    community_labels = [f"{cid} — {comm_names[cid]}" for cid in sorted(comm_names)]
    filtered_labels = [
        label
        for label in community_labels
        if query.lower() in label.lower() or query == ""
    ]
    selected_label = st.selectbox("Выберите вашу тематику:", filtered_labels)
    selected_comm_id = int(selected_label.split(" — ")[0])

    # --- Keyword selection within the chosen community ---
    keywords = get_keywords_by_comm(df, selected_comm_id)
    selected_keywords = st.multiselect(
        "Выберите ключевые слова по теме:", keywords
    )

    if "show_collab" not in st.session_state:
        st.session_state.show_collab = False

    # Only allow search when at least one keyword is selected
    if selected_keywords:
        if st.button("Найти возможные тематики для коллаборации"):
            st.session_state.show_collab = True

    if st.session_state.show_collab:
        # --- Find suggested communities and display them ---
        comm_ids, related_keywords = find_suggested_communities(
            df, selected_comm_id, selected_keywords
        )
        if comm_ids:
            st.markdown("### Возможные тематики для коллаборации:")
            collab_labels = [
                f"{cid} — {comm_names.get(cid, '')}"
                for cid in comm_ids
            ]
            for cid in comm_ids:
                label = f"{cid} — {comm_names.get(cid, '')}"
                top_related = ", ".join(
                    sorted(list(related_keywords[cid]))[:5]
                )
                st.write(f"**{label}**: {top_related}")

            # Allow user to choose one of the suggested communities
            selected_collab_label = st.selectbox(
                "Выберите тематику для коллаборации:", collab_labels
            )
            selected_collab_id = int(selected_collab_label.split(" — ")[0])
            # Keywords for secondary community
            collab_keywords = get_keywords_by_comm(df, selected_collab_id)
            selected_collab_keywords = st.multiselect(
                "Выберите ключевые слова по второй теме:", collab_keywords
            )
            if selected_collab_keywords:
                # Compute recommended authors
                authors = find_recommended_authors(
                    df,
                    selected_comm_id,
                    selected_keywords,
                    selected_collab_id,
                    selected_collab_keywords,
                )
                if authors:
                    st.markdown("### Рекомендуемые авторы:")
                    for author, kwds in authors:
                        st.write(f"{author} — ", ", ".join(kwds))
                else:
                    st.warning(
                        "Не найдено авторов, публикующихся по обеим темам с указанными ключевыми словами."
                    )
        else:
            st.warning(
                "Не найдены возможные тематики для коллаборации с выбранными ключевыми словами."
            )


if __name__ == "__main__":
    main()
