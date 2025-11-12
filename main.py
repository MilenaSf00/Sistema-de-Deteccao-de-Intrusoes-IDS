import pandas as pd
import numpy as np

# Defina o nome do seu arquivo
file_path = 'Monday-WorkingHours.pcap_ISCX.csv'

try:
    # 1. Carregar o dataset
    # Usamos 'low_memory=False' para evitar avisos sobre tipos de dados mistos
    print(f"Carregando o arquivo: {file_path} ...")
    df = pd.read_csv(file_path, low_memory=False)

    # 2. Limpeza de Dados Inválidos (Muito importante para este dataset)
    
    # Remove espaços em branco dos nomes das colunas (problema comum)
    df.columns = df.columns.str.strip()

    # Passo 2a: Substituir 'Infinity' e '-Infinity' por 'NaN'
    # Isso ocorre em colunas como 'Flow Bytes/s' e 'Flow Packets/s'
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Passo 2b: Remover todas as linhas que contenham qualquer valor NaN
    # Esta é uma abordagem. Outra seria preencher com a média/mediana.
    print(f"Formato original: {df.shape}")
    df.dropna(inplace=True)
    print(f"Formato após remover NaNs: {df.shape}")

    # 3. Lidar com colunas duplicadas (Ex: 'Fwd Header Length')
    # O .loc[:,~df.columns.duplicated()] seleciona todas as colunas
    # que NÃO são duplicadas, mantendo a primeira ocorrência.
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 4. Verificação Rápida
    print("\n--- Informações do DataFrame Limpo ---")
    df.info()

    # 5. Contagem dos Rótulos (Labels)
    # Isso é crucial para saber se o dataset é balanceado
    print("\n--- Contagem de Rótulos (Labels) ---")
    print(df['Label'].value_counts())
    
    # Salvar o dataframe limpo (opcional)
    # df.to_csv('cleaned_monday.csv', index=False)


except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro: {e}")