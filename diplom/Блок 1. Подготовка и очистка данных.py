import pandas as pd
import re

# Функция очистки колонки Abstract
def clean_abstract_column(df: pd.DataFrame) -> pd.DataFrame:
    column = 'Abstract'
    df = df.drop_duplicates(subset=[column]).dropna(subset=[column])
    df = df[df[column].str.strip().str.len() > 10]
    df[column] = df[column].apply(lambda x: re.sub(r'<.*?>', '', x))
    df = df[~df[column].str.contains(r'<jats:p>|&lt;', regex=True)]
    df[column] = df[column].apply(lambda x: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', x))
    return df

# Основная функция обработки данных
def process_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    required_columns = ['Author/s', 'Abstract', 'Source Country', 'Keywords']
    df = df[required_columns]
    df.dropna(subset=['Author/s', 'Source Country'], inplace=True)
    df = df[~(df['Abstract'].isna() & df['Keywords'].isna())]
    df.drop_duplicates(inplace=True)
    df = clean_abstract_column(df)
    if df.empty:
        raise ValueError("После очистки данных DataFrame пуст. Проверьте исходные данные.")
    return df

# Указываем путь к данным
input_file_path = r"E:\delete\lens-export.csv"
cleaned_df = process_data(input_file_path)
