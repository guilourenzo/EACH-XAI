import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import sys
import json

# Diretório onde os arquivos CSV estão localizados
diretorio = 'public/dataset/'

# Lista todos os arquivos no diretório
arquivos = os.listdir(diretorio)

# Verifica se há algum arquivo CSV na lista
arquivos_csv = [arquivo for arquivo in arquivos if arquivo.endswith('.csv')]

# Se houver pelo menos um arquivo CSV, lê o primeiro arquivo encontrado
if arquivos_csv:
    arquivo_csv = arquivos_csv[0]
    caminho_arquivo = os.path.join(diretorio, arquivo_csv)

    # Realiza a leitura do arquivo CSV
    df = pd.read_csv(caminho_arquivo).dropna()

    # Arquivo para salvar os mapeamentos
    with open('public/data_preprocess/preprocess_dataset.txt', 'w') as f:
        f.write("Mapeamentos do LabelEncoder para variáveis qualitativas:\n\n")

        # Aplica LabelEncoder nas colunas qualitativas
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

            # Salva o mapeamento no arquivo de texto
            f.write(f"Coluna: {col}\n")
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            for original, encoded in mapping.items():
                f.write(f"  {original}: {encoded}\n")
            f.write("\n")

    # Prepara a saída com número de variáveis e lista de variáveis
    output = {
        "num_variaveis": max(0, df.shape[1] - 1),
        "variaveis": df.columns.tolist()
    }

    # Escreve a saída no stdout como JSON
    sys.stdout.write(json.dumps(output))
else:
    raise FileNotFoundError(
        "Nenhum arquivo CSV encontrado no diretório especificado.")
