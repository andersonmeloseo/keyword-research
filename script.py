import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from urllib.parse import urlparse

def load_data(excel_file='kw.xlsx'):
    """
    Lê o arquivo Excel contendo as colunas:
    Keyword, Intent, Volume, Trend, Keyword Difficulty,
    CPC (USD), Competitive Density, SERP Features
    Retorna um DataFrame pandas.
    """
    df = pd.read_excel(excel_file)
    
    # Limpeza inicial: remover linhas totalmente vazias
    df.dropna(how='all', inplace=True)
    
    # Opcional: remover duplicadas
    df.drop_duplicates(subset=['Keyword'], inplace=True)
    
    # Garantir que colunas numéricas estejam no tipo correto
    numeric_cols = ['Volume', 'Trend', 'Keyword Difficulty', 'CPC (USD)', 'Competitive Density']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Preencher NaN em colunas de string com vazio
    df.fillna({'Keyword': '', 'Intent': '', 'SERP Features': ''}, inplace=True)
    
    return df

def create_summary_stats(df):
    """
    Gera um DataFrame com estatísticas gerais:
    - Contagem total de linhas
    - Média, mediana, máx, mín de Volume, CPC, etc.
    """
    # Exemplo de colunas que analisaremos: Volume, CPC (USD), Keyword Difficulty
    summary_data = []
    
    total_keywords = len(df)
    summary_data.append(['Total de Palavras-Chave', total_keywords])
    
    # Volume
    if 'Volume' in df.columns:
        summary_data.append(['Volume (Média)', df['Volume'].mean(skipna=True)])
        summary_data.append(['Volume (Mediana)', df['Volume'].median(skipna=True)])
        summary_data.append(['Volume (Máx)', df['Volume'].max(skipna=True)])
        summary_data.append(['Volume (Mín)', df['Volume'].min(skipna=True)])
    
    # CPC (USD)
    if 'CPC (USD)' in df.columns:
        summary_data.append(['CPC (Média)', df['CPC (USD)'].mean(skipna=True)])
        summary_data.append(['CPC (Mediana)', df['CPC (USD)'].median(skipna=True)])
        summary_data.append(['CPC (Máx)', df['CPC (USD)'].max(skipna=True)])
        summary_data.append(['CPC (Mín)', df['CPC (USD)'].min(skipna=True)])
    
    # Keyword Difficulty
    if 'Keyword Difficulty' in df.columns:
        summary_data.append(['Keyword Difficulty (Média)', df['Keyword Difficulty'].mean(skipna=True)])
        summary_data.append(['Keyword Difficulty (Mediana)', df['Keyword Difficulty'].median(skipna=True)])
        summary_data.append(['Keyword Difficulty (Máx)', df['Keyword Difficulty'].max(skipna=True)])
        summary_data.append(['Keyword Difficulty (Mín)', df['Keyword Difficulty'].min(skipna=True)])
    
    summary_df = pd.DataFrame(summary_data, columns=['Métrica', 'Valor'])
    return summary_df

def create_semantic_groups(df, n_clusters=5):
    """
    Cria grupos semânticos a partir da coluna 'Keyword' usando TF-IDF e KMeans.
    Adiciona a coluna 'SemanticGroup' ao DataFrame e retorna-o.
    """
    if 'Keyword' not in df.columns or df['Keyword'].isna().all():
        df['SemanticGroup'] = -1
        return df
    
    keywords = df['Keyword'].astype(str).tolist()
    
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    X = vectorizer.fit_transform(keywords)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    df['SemanticGroup'] = labels
    return df

def group_by_intent(df):
    """
    Ordena o DataFrame por 'Intent' e retorna esse DataFrame.
    """
    if 'Intent' in df.columns:
        df_sorted = df.sort_values(by='Intent', na_position='last')
    else:
        df_sorted = df.copy()
    return df_sorted

def filter_questions(df):
    """
    Filtra as palavras que são perguntas.
    Critérios simples:
      - Contêm '?' OU
      - Começam com tokens típicos de pergunta (ex.: como, quando, por que, qual, quais, etc.)
    """
    question_words = ['como', 'quando', 'por que', 'porquê', 'qual', 'quais', 'onde', 'quem']  # Ajuste livre
    def is_question(kw):
        kw_lower = kw.lower()
        if '?' in kw_lower:
            return True
        for w in question_words:
            if kw_lower.startswith(w + ' '):
                return True
        return False
    
    if 'Keyword' not in df.columns:
        return pd.DataFrame(columns=df.columns)
    
    df_questions = df[df['Keyword'].astype(str).apply(is_question)]
    return df_questions

def filter_local_pack(df):
    """
    Filtra as palavras que acionam a SERP Feature 'Local pack'.
    Verifica se a coluna 'SERP Features' contém a substring 'Local pack'.
    """
    if 'SERP Features' not in df.columns:
        return pd.DataFrame(columns=df.columns)
    
    return df[df['SERP Features'].astype(str).str.contains('Local pack', case=False, na=False)]

def main():
    # 1. Ler o arquivo kw.xlsx e limpar
    df = load_data('kw.xlsx')
    
    # 2. Gerar estatísticas gerais (para a aba "Resumo e Estatísticas")
    summary_df = create_summary_stats(df)
    
    # 3. Criar grupos semânticos (aba "Grupos Semanticos")
    df_semantic = create_semantic_groups(df.copy(), n_clusters=5)
    
    # 4. Agrupar por intenção (aba "Intencao de Busca")
    df_intent = group_by_intent(df.copy())
    
    # 5. Filtrar perguntas (aba "Perguntas")
    df_questions = filter_questions(df.copy())
    
    # 6. Local pack (aba "Local Pack")
    df_localpack = filter_local_pack(df.copy())
    
    # 7. Gerar arquivo Excel de saída
    output_file = 'kw_analysis_output.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Aba Resumo e Estatísticas
        summary_df.to_excel(writer, sheet_name='Resumo e Estatisticas', index=False)
        
        # Aba Grupos Semanticos
        df_semantic.to_excel(writer, sheet_name='Grupos Semanticos', index=False)
        
        # Aba Intencao de Busca
        df_intent.to_excel(writer, sheet_name='Intencao de Busca', index=False)
        
        # Aba Perguntas
        df_questions.to_excel(writer, sheet_name='Perguntas', index=False)
        
        # Aba Local Pack
        df_localpack.to_excel(writer, sheet_name='Local Pack', index=False)
    
    print(f"Análise concluída com sucesso!\nArquivo gerado: {os.path.abspath(output_file)}")

if __name__ == '__main__':
    main()
