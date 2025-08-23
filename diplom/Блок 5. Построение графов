import pandas as pd
import networkx as nx
from itertools import combinations
from collections import defaultdict, Counter
import ast

# Загружаем подготовленный датасет
df = pd.read_csv("Big_data.csv")

# Подготовка данных

def safe_literal_eval(value):
    """Безопасное приведение строковых представлений списков к объектам."""
    try:
        return ast.literal_eval(value) if isinstance(value, str) else value
    except:
        return []

# Преобразуем ключевые колонки
df['Normalized Keywords'] = df['Normalized Keywords'].apply(safe_literal_eval)
df['Extracted Keywords'] = df['Extracted Keywords'].apply(safe_literal_eval)

# Обрабатываем авторов
def process_authors(value):
    if isinstance(value, str):
        return [a.strip() for a in value.split(';') if a.strip()]
    return []

df['Authors'] = df['Author/s'].fillna('').apply(process_authors)

# Обрабатываем ключевые слова из колонки Keywords
def process_keywords(value):
    if isinstance(value, str):
        return [k.strip().lower() for k in value.split(';') if k.strip()]
    return []

if 'Keywords' in df.columns:
    df['Keywords'] = df['Keywords'].fillna('').apply(process_keywords)

# Граф соавторства (Co-Authorship Graph)

G_auth = nx.Graph()

for authors in df['Authors']:
    if not isinstance(authors, list) or len(authors) < 2:
        continue
    clean_authors = list(set(a.strip() for a in authors if a.strip()))
    for a1, a2 in combinations(clean_authors, 2):
        if G_auth.has_edge(a1, a2):
            G_auth[a1][a2]['weight'] += 1
        else:
            G_auth.add_edge(a1, a2, weight=1)

G_auth.remove_nodes_from([n for n in G_auth.nodes if not n.strip()])
nx.write_graphml(G_auth, "graph_coauthorship.graphml")


# Граф ключевых слов из колонки Keywords

G_kw = nx.Graph()

for keywords in df['Keywords']:
    for k1, k2 in combinations(set(keywords), 2):
        G_kw.add_edge(k1, k2)

nx.write_graphml(G_kw, "graph_keywords_column.graphml")

# Граф взаимодействия тематических сообществ

def process_communities(value):
    try:
        if isinstance(value, str):
            return ast.literal_eval(value)
        elif isinstance(value, list):
            return value
        else:
            return []
    except:
        return []

if 'Keyword Communities' in df.columns:
    df['Keyword Communities'] = df['Keyword Communities'].apply(process_communities)

keyword_to_community = {}
community_keywords = defaultdict(list)

for community in df['Keyword Communities']:
    for keyword, comm_id in community:
        keyword_to_community[keyword] = comm_id
        community_keywords[comm_id].append(keyword)

community_names = {
    comm_id: ", ".join(keywords[:3])
    for comm_id, keywords in community_keywords.items()
}

G_meta = nx.Graph()

for keywords in df['Normalized Keywords']:
    communities_in_article = set()
    for kw in keywords:
        if kw in keyword_to_community:
            comm_id = keyword_to_community[kw]
            label = community_names.get(comm_id, str(comm_id))
            communities_in_article.add(label)
    for c1, c2 in combinations(communities_in_article, 2):
        if G_meta.has_edge(c1, c2):
            G_meta[c1][c2]['weight'] += 1
        else:
            G_meta.add_edge(c1, c2, weight=1)

nx.write_graphml(G_meta, "graph_community_interactions_named.graphml")

# Граф топ-10000 нормализованных ключевых слов

all_keywords = [kw for keywords in df['Normalized Keywords'] for kw in keywords]
keyword_counts = Counter(all_keywords)
top_keywords = set([kw for kw, _ in keyword_counts.most_common(10000)])

G_nkw_top10000 = nx.Graph()

for keywords in df['Normalized Keywords']:
    filtered = [kw for kw in keywords if kw in top_keywords]
    for k1, k2 in combinations(set(filtered), 2):
        G_nkw_top10000.add_edge(k1, k2)

nx.write_graphml(G_nkw_top10000, "graph_normalized_keywords_top10000.graphml")

# Двудольный граф (Bipartite Graph)

B_top10000 = nx.Graph()

for _, row in df.iterrows():
    authors = row['Authors']
    keywords = [kw for kw in row['Normalized Keywords'] if kw in top_keywords]
    for a in authors:
        B_top10000.add_node(a, bipartite=0)
    for kw in keywords:
        B_top10000.add_node(kw, bipartite=1)
    for a in authors:
        for kw in keywords:
            B_top10000.add_edge(a, kw)

nx.write_graphml(B_top10000, "graph_competence_bipartite_top10000.graphml")
