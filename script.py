import os
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def load_data(excel_file='kw.xlsx'):
    """
    Lê o arquivo Excel 'kw.xlsx' e retorna um DataFrame.
    O arquivo deve conter as colunas:
    Keyword, Intent, Volume, Trend, Keyword Difficulty, CPC (USD),
    Competitive Density, SERP Features.
    """
    df = pd.read_excel(excel_file)
    df.dropna(how='all', inplace=True)
    df.drop_duplicates(subset=['Keyword'], inplace=True)
    
    numeric_cols = ['Volume', 'Trend', 'Keyword Difficulty', 'CPC (USD)', 'Competitive Density']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.fillna({'Keyword': '', 'Intent': '', 'SERP Features': ''}, inplace=True)
    return df

def create_summary_stats(df):
    """
    Gera um DataFrame com estatísticas gerais (total de keywords, média de Volume, etc.).
    """
    summary_data = []
    total_keywords = len(df)
    summary_data.append(['Total de Palavras-Chave', total_keywords])
    if 'Volume' in df.columns:
        summary_data.append(['Volume (Média)', df['Volume'].mean(skipna=True)])
    if 'CPC (USD)' in df.columns:
        summary_data.append(['CPC (Média)', df['CPC (USD)'].mean(skipna=True)])
    if 'Keyword Difficulty' in df.columns:
        summary_data.append(['Keyword Difficulty (Média)', df['Keyword Difficulty'].mean(skipna=True)])
    
    return pd.DataFrame(summary_data, columns=['Métrica', 'Valor'])

def create_semantic_groups(df, n_clusters=5):
    """
    Agrupa semanticamente as palavras-chave usando TF-IDF e KMeans.
    Adiciona uma coluna 'SemanticGroup' ao DataFrame.
    """
    if 'Keyword' not in df.columns or df['Keyword'].isna().all():
        df['SemanticGroup'] = -1
        return df
    keywords = df['Keyword'].astype(str).tolist()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    X = vectorizer.fit_transform(keywords)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    df['SemanticGroup'] = kmeans.labels_
    return df

def group_by_intent(df):
    """
    Ordena o DataFrame pela coluna 'Intent' e retorna-o.
    """
    if 'Intent' in df.columns:
        return df.sort_values(by='Intent', na_position='last')
    return df.copy()

def filter_questions(df):
    """
    Filtra as palavras-chave que parecem perguntas em Português, Inglês ou Espanhol.
    Consideramos termos como 'como', 'quando', 'por que', 'what', 'when', etc., ou contendo '?'.
    """
    question_words = [
        # Português
        'como', 'quando', 'por que', 'porquê', 'qual', 'quais', 'onde', 'quem',
        # Inglês
        'what', 'when', 'where', 'why', 'how', 'who', 'which', 'whose',
        # Espanhol
        'cómo', 'cuándo', 'por qué', 'porqué', 'cuál', 'cuáles', 'dónde', 'quién'
    ]
    def is_question(kw):
        kw_lower = kw.lower()
        if '?' in kw_lower:
            return True
        for w in question_words:
            # Verifica se começa com a palavra + espaço (ex.: "como fazer...")
            if kw_lower.startswith(w + ' '):
                return True
        return False
    
    if 'Keyword' not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df['Keyword'].astype(str).apply(is_question)]

def filter_local_pack(df):
    """
    Filtra as palavras-chave cuja coluna 'SERP Features' contenha 'Local pack'.
    """
    if 'SERP Features' not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df['SERP Features'].astype(str).str.contains('Local pack', case=False, na=False)]

def create_intent_sheet_dfs(df):
    """
    Cria um dicionário de DataFrames, um para cada valor único da coluna 'Intent'.
    """
    sheets = {}
    if 'Intent' in df.columns:
        for intent in df['Intent'].dropna().unique():
            sheets[str(intent)] = df[df['Intent'] == intent]
    return sheets

def create_cluster_sheet_dfs(df):
    """
    Cria um dicionário de DataFrames, um para cada valor único da coluna 'SemanticGroup'.
    """
    sheets = {}
    if 'SemanticGroup' in df.columns:
        for cluster in df['SemanticGroup'].unique():
            sheets[f'Cluster {cluster}'] = df[df['SemanticGroup'] == cluster]
    return sheets

def write_excel_report(df, output_filename, n_clusters=5):
    """
    Gera o relatório Excel contendo:
    - Resumo e Estatísticas
    - Grupos Semânticos
    - Intenção de Busca
    - Perguntas
    - Local Pack
    - Abas individuais para cada valor de 'Intent'
    - Abas individuais para cada valor de 'SemanticGroup'
    """
    summary_df = create_summary_stats(df)
    semantic_df = create_semantic_groups(df.copy(), n_clusters=n_clusters)
    intent_df = group_by_intent(df.copy())
    questions_df = filter_questions(df.copy())
    localpack_df = filter_local_pack(df.copy())
    
    intent_sheets = create_intent_sheet_dfs(df)
    cluster_sheets = create_cluster_sheet_dfs(semantic_df)
    
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Resumo e Estatísticas', index=False)
        semantic_df.to_excel(writer, sheet_name='Grupos Semânticos', index=False)
        intent_df.to_excel(writer, sheet_name='Intenção de Busca', index=False)
        questions_df.to_excel(writer, sheet_name='Perguntas', index=False)
        localpack_df.to_excel(writer, sheet_name='Local Pack', index=False)
        
        # Abas individuais para cada intenção
        for intent_value, df_intent in intent_sheets.items():
            sheet_name = f"Intent - {intent_value}"
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            df_intent.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Abas individuais para cada cluster semântico
        for cluster_name, df_cluster in cluster_sheets.items():
            sheet_name = cluster_name
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            df_cluster.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Relatório gerado: {os.path.abspath(output_filename)}")

def main():
    df = load_data('kw.xlsx')
    data_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")  # somente data e hora
    output_filename = f"{data_str}.xlsx"
    write_excel_report(df, output_filename, n_clusters=5)

if __name__ == '__main__':
    main()
