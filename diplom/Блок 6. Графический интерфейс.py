import streamlit as st
import pandas as pd
import ast
from collections import defaultdict, Counter

# Настройка интерфейса

st.title("Система поиска научного сотрудничества")

# Загрузка данных

df = pd.read_csv("Big_data.csv")
df.rename(columns={'Author/s': 'Author_s'}, inplace=True)

def safe_eval(val):
    """Безопасная конвертация строки в список"""
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except:
        return []

df['Author_s'] = df['Author_s'].fillna('Неизвестный')
df.rename(columns={"Keyword Communities": "Keyword_Communities"}, inplace=True)
df['Normalized Keywords'] = df['Normalized Keywords'].apply(safe_eval)
df['Keyword_Communities'] = df['Keyword_Communities'].apply(safe_eval)

# Формирование топ-3 ключевых слов для каждого сообщества

comm_to_words = defaultdict(list)
for row in df['Keyword_Communities']:
    for word, comm_id in row:
        comm_to_words[comm_id].append(word)

comm_names = {
    comm_id: ", ".join([kw for kw, _ in Counter(words).most_common(3)])
    for comm_id, words in comm_to_words.items()
}

# Поиск тематических сообществ

query = st.text_input("Введите ключевое слово для поиска тематики")

community_labels = [
    f"{comm_id} — {comm_names[comm_id]}"
    for comm_id in sorted(comm_names)
]

filtered_labels = [
    label for label in community_labels
    if query.lower() in label.lower() or query == ""
]

selected_label = st.selectbox("Выберите тематику:", filtered_labels)
selected_comm_id = int(selected_label.split(" — ")[0])

# Выбор ключевых слов для выбранной тематики

def get_keywords_by_community(comm_id):
    keywords = set()
    for row in df['Keyword_Communities']:
        for kw, cid in row:
            if cid == comm_id:
                keywords.add(kw)
    return sorted(list(keywords))

selected_keywords = st.multiselect(
    "Выберите ключевые слова по теме:",
    get_keywords_by_community(selected_comm_id)
)

# Поиск тематик для коллаборации

if 'show_collab' not in st.session_state:
    st.session_state.show_collab = False

if selected_keywords:
    if st.button("Найти возможные тематики для коллаборации"):
        st.session_state.show_collab = True

if st.session_state.show_collab:
    suggested_communities = set()
    related_keywords = defaultdict(set)

    for _, row in df.iterrows():
        if all((kw, selected_comm_id) in row['Keyword_Communities'] for kw in selected_keywords):
            for other_kw, other_comm_id in row['Keyword_Communities']:
                if other_comm_id != selected_comm_id:
                    suggested_communities.add(other_comm_id)
                    related_keywords[other_comm_id].add(other_kw)

    # Выбор темы для коллаборации

    if suggested_communities:
        collab_labels = [
            f"{cid} — {comm_names.get(cid, str(cid))}"
            for cid in sorted(suggested_communities)
        ]
        selected_collab_label = st.selectbox(
            "Выберите тематику для коллаборации:",
            collab_labels
        )
        collab_comm_id = int(selected_collab_label.split(" — ")[0])

        selected_collab_keywords = st.multiselect(
            "Выберите ключевые слова по второй теме:",
            sorted(list(related_keywords[collab_comm_id]))
        )

        # Поиск авторов по выбранным темам

        if selected_collab_keywords:
            unique_authors = set()

            for row in df.itertuples():
                comms = getattr(row, 'Keyword_Communities')
                has_main = all((kw, selected_comm_id) in comms for kw in selected_keywords)
                has_collab = all((kw, collab_comm_id) in comms for kw in selected_collab_keywords)

                if has_main and has_collab:
                    authors_raw = getattr(row, 'Author_s')
                    for author in str(authors_raw).split(";"):
                        cleaned = author.strip()
                        if cleaned and cleaned.lower() != "null null":
                            unique_authors.add(cleaned)

            # Вывод найденных авторов

            if unique_authors:
                st.markdown("###  Рекомендуемые авторы:")
                for author in sorted(unique_authors):
                    author_rows = df[df['Author_s'].str.contains(author, na=False)]
                    author_topics = set()
                    for kw_list in author_rows['Keyword_Communities']:
                        for _, cid in kw_list:
                            author_topics.add(comm_names.get(cid, str(cid)))
                    topics_str = ", ".join(sorted(author_topics))
                    st.markdown(f"**{author}** — _{topics_str}_")

                # Отображение публикаций для авторов

                lens_df = pd.read_csv("lens-export.csv")
                lens_df.fillna("-", inplace=True)

                tabs = st.tabs(sorted(unique_authors))
                for tab, author in zip(tabs, sorted(unique_authors)):
                    with tab:
                        pub_data = lens_df[lens_df['Author/s'].str.contains(author, na=False, case=False)]
                        if not pub_data.empty:
                            st.dataframe(pub_data[['Title', 'Abstract', 'Source Country', 'Fields of Study']])
                        else:
                            st.info("Нет публикаций для отображения")
            else:
                st.warning(" Не найден автор, объединяющий выбранные темы")
        else:
            st.info(" Выберите хотя бы одно ключевое слово по второй теме")
    else:
        st.warning(" Не найдены возможные тематики для коллаборации")
