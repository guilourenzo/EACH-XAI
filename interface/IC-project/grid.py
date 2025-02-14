import os
import pandas as pd
import optuna
import re
import sys
import json
import joblib
import numpy as np
import warnings
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

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
else:
    raise FileNotFoundError(
        "Nenhum arquivo CSV encontrado no diretório especificado.")


# Função para carregar o arquivo e definir as variáveis
def load_grid_config(file_path):
    global scalers, num_folds, min_k, max_k, score_functions, activation_functions, solvers, learning_rates, num_trials
    global max_learning_rate_init, min_learning_rate_init, min_alfa, max_alfa, hidden_layer_size, max_epochs, min_epochs
    global target_columns, deleted_columns

    num_folds = None
    min_k = None
    max_k = None
    score_functions = []
    activation_functions = []
    solvers = []
    learning_rates = []
    max_learning_rate_init = None
    min_learning_rate_init = None
    min_alfa = None
    max_alfa = None
    hidden_layer_size = []
    max_epochs = None
    min_epochs = None
    num_trials = None
    target_columns = []
    deleted_columns = []
    scalers = []

    # Regex
    patterns = {
        "NumFolds": r"NumFolds:\s*(\d+)",
        "minK": r"minK:\s*(\d+)",
        "maxK": r"maxK:\s*(\d+)",
        "Score Function": r"Score Function:\s*([\w, ]+)",
        "Activation Function": r"Activation Function:\s*([\w, ]+)",
        "Solver": r"Solver:\s*([\w, ]+)",
        "Learning Rate": r"Learning Rate:\s*([\w, ]+)",
        "maxLearningRateInit": r"maxLearningRateInit:\s*([\d.]+)",
        "minLearningRateInit": r"minLearningRateInit:\s*([\d.]+)",
        "minAlfa": r"minAlfa:\s*([\d.]+)",
        "maxAlfa": r"maxAlfa:\s*([\d.]+)",
        "Hidden Layer Sizes": r"Hidden Layer Sizes:\s*([\(\d+(\s*,\s*\d+)*\)]+)",
        "maxEpochs": r"maxEpochs:\s*(\d+)",
        "minEpochs": r"minEpochs:\s*(\d+)",
        "NumTrials": r"NumTrials:\s*(\d+)",
        "Var Target": r"Var Target:\s*([\w, ]*)(?=\nVar Deleted|$)",
        "Var Deleted": r"Var Deleted:\s*([\w, ]*)(?=\nScalers|$)",
        "Scalers": r"Scalers:\s*([\w, ]+)",
    }

    with open(file_path, 'r') as file:
        content = file.read()

        # Usando regex para capturar os valores
        num_folds = int(re.search(patterns["NumFolds"], content).group(
            1)) if re.search(patterns["NumFolds"], content) else None

        min_k = int(re.search(patterns["minK"], content).group(
            1)) if re.search(patterns["minK"], content) else None
        max_k = int(re.search(patterns["maxK"], content).group(
            1)) if re.search(patterns["maxK"], content) else None

        num_trials = int(re.search(patterns["NumTrials"], content).group(
            1)) if re.search(patterns["NumTrials"], content) else None

        # Função de Score Function
        score_function_values = re.search(patterns["Score Function"], content)
        if score_function_values:
            score_functions = [
                func.strip() for func in score_function_values.group(1).split(",")]

        # Funções de Activation Function
        activation_function_values = re.search(
            patterns["Activation Function"], content)
        if activation_function_values:
            activation_functions = [
                func.strip() for func in activation_function_values.group(1).split(",")]

        # Solvers
        solver_values = re.search(patterns["Solver"], content)
        if solver_values:
            solvers = [solver.strip()
                       for solver in solver_values.group(1).split(",")]

        # Learning Rates
        learning_rate_values = re.search(patterns["Learning Rate"], content)
        if learning_rate_values:
            learning_rates = [rate.strip()
                              for rate in learning_rate_values.group(1).split(",")]

        # Valores numéricos
        max_learning_rate_init = float(re.search(patterns["maxLearningRateInit"], content).group(
            1)) if re.search(patterns["maxLearningRateInit"], content) else None
        min_learning_rate_init = float(re.search(patterns["minLearningRateInit"], content).group(
            1)) if re.search(patterns["minLearningRateInit"], content) else None
        min_alfa = float(re.search(patterns["minAlfa"], content).group(
            1)) if re.search(patterns["minAlfa"], content) else None
        max_alfa = float(re.search(patterns["maxAlfa"], content).group(
            1)) if re.search(patterns["maxAlfa"], content) else None
        min_epochs = int(re.search(patterns["minEpochs"], content).group(
            1)) if re.search(patterns["minEpochs"], content) else None
        max_epochs = int(re.search(patterns["maxEpochs"], content).group(
            1)) if re.search(patterns["maxEpochs"], content) else None

        # Hidden Layer Sizes
        hidden_layer_values = re.search(
            patterns["Hidden Layer Sizes"], content)
        if hidden_layer_values:
            for size in hidden_layer_values.group(1).split("),"):
                # Remover espaços extras e parênteses
                clean_size = re.sub(r'[()]', '', size).strip()
                # Se houver mais de um valor, armazene como tupla
                if ',' in clean_size:
                    hidden_layer_size.append(
                        tuple(map(int, clean_size.split(','))))
                else:
                    # Como tupla com um único valor
                    hidden_layer_size.append((int(clean_size),))

        # Captura da variável alvo e as excluídas
        target_columns_values = re.search(
            patterns["Var Target"], content)
        if target_columns_values:
            target_columns = [target_col.strip()
                              for target_col in target_columns_values.group(1).split(",")]
        target_columns = [
            target_col for target_col in target_columns if target_col != '']

        deleted_columns_values = re.search(
            patterns["Var Deleted"], content)
        if deleted_columns_values:
            deleted_columns = [
                deleted_col.strip() for deleted_col in deleted_columns_values.group(1).split(",")]
        deleted_columns = [
            deleted_col for deleted_col in deleted_columns if deleted_col != '']

        # Escalonadores
        scaler_values = re.search(
            patterns["Scalers"], content)
        if scaler_values:
            scalers = [scaler.strip()
                       for scaler in scaler_values.group(1).split(",")]


# Caminho do arquivo de configuração
file_path = sys.argv[1]

# Carregar as variáveis do arquivo
load_grid_config(file_path)

# Divisão do dataset
X_df = df.drop(columns=[target_columns[0]])
y_df = df[target_columns[0]]

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42)

if y_df.dtype == 'object' or y_df.dtype.name == 'category':
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

max_k = min(max_k, X_train.shape[1] - len(deleted_columns))

# Suprime os avisos específicos do Optuna
warnings.filterwarnings(
    "ignore", message="Choices for a categorical distribution should be a tuple of None, bool, int, float and str")


def objective(trial):
    # Hiperparâmetro para o escalonador
    scaler_option = trial.suggest_categorical('scaler', scalers)

    # Escolha do escalonador baseado no hiperparâmetro
    if scaler_option == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_option == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_option == 'RobustScaler':
        scaler = RobustScaler()

    # Hiperparâmetros do SelectKBest
    k = trial.suggest_int('kbest__k', min_k, max_k)
    score_func_option = trial.suggest_categorical(
        'kbest__score_func', score_functions)

    if score_func_option == 'f_classif':
        score_func = f_classif
    elif score_func_option == 'chi2':
        score_func = chi2
    elif score_func_option == 'mutual_info_classif':
        score_func = mutual_info_classif

    # Hiperparâmetros do MLP
    hidden_layer_sizes = trial.suggest_categorical(
        'mlp__hidden_layer_sizes', hidden_layer_size)
    alpha = trial.suggest_float('mlp__alpha', min_alfa, max_alfa, log=True)
    learning_rate_init = trial.suggest_float(
        'mlp__learning_rate_init', min_learning_rate_init, max_learning_rate_init, log=True)
    activation = trial.suggest_categorical(
        'mlp__activation', activation_functions)
    solver = trial.suggest_categorical('mlp__solver', solvers)
    learning_rate = trial.suggest_categorical(
        'mlp__learning_rate', learning_rates)
    max_iter = trial.suggest_int(
        'mlp__max_iter', min_epochs, max_epochs)

    # Configuração do pipeline
    preprocessor = ColumnTransformer([
        ('deleter', 'drop', deleted_columns),
        ('scaler', scaler, [
         col for col in X_train.columns if col not in deleted_columns])
    ])
    pipeline = SKPipeline([
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer(strategy='mean')),
        ('kbest', SelectKBest(score_func=score_func, k=k)),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            learning_rate=learning_rate,
        ))
    ])

    # Validação cruzada com tratamento de erro para chi2
    try:
        cv = StratifiedKFold(n_splits=num_folds, shuffle=True)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                 scoring='accuracy', n_jobs=-1)
        return scores.mean()
    except ValueError as e:
        # Ignora erros do chi2 relacionados a valores negativos
        if 'Input X must be non-negative' in str(e):
            return float('-inf')  # Indica falha no trial
        else:
            raise e  # Levanta outros erros para depuração


# Estudo com Optuna
study = optuna.create_study(direction='maximize')

# Número de combinações a testar
study.optimize(objective, n_trials=num_trials)

# Configurando o pipeline com os melhores hiperparâmetros encontrados
best_params = study.best_params

# Inicializando o pipeline com os melhores parâmetros
scaler_option = best_params['scaler']
if scaler_option == 'MinMaxScaler':
    scaler = MinMaxScaler()
elif scaler_option == 'StandardScaler':
    scaler = StandardScaler()
elif scaler_option == 'RobustScaler':
    scaler = RobustScaler()

score_func_option = best_params["kbest__score_func"]
if score_func_option == 'f_classif':
    score_func = f_classif
elif score_func_option == 'chi2':
    score_func = chi2
elif score_func_option == 'mutual_info_classif':
    score_func = mutual_info_classif

preprocessor = ColumnTransformer([
    ('deleter', 'drop', deleted_columns),
    ('scaler', scaler, [
     col for col in X_train.columns if col not in deleted_columns])
])

pipeline_final = SKPipeline([
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),
    ('kbest', SelectKBest(
        score_func=score_func, k=best_params['kbest__k'])),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=best_params['mlp__hidden_layer_sizes'],
        alpha=best_params['mlp__alpha'],
        learning_rate_init=best_params['mlp__learning_rate_init'],
        activation=best_params['mlp__activation'],
        solver=best_params['mlp__solver'],
        max_iter=best_params['mlp__max_iter'],
        learning_rate=best_params['mlp__learning_rate'],
        random_state=42
    ))
])

# Treinando o modelo
pipeline_final.fit(X_train, y_train)

y_pred = pipeline_final.predict(X_test)

# Obter a curva de perda do MLP
mlp_model = pipeline_final.named_steps['mlp']
loss_curve = mlp_model.loss_curve_


# Resultados
def serialize_best_params(best_params):
    serializable_params = {}
    for key, value in best_params.items():
        if callable(value):
            serializable_params[key] = value.__name__
        else:
            serializable_params[key] = value
    return serializable_params


result = {
    "best_params": serialize_best_params(study.best_params),
    "best_value": study.best_value
}

joblib.dump(pipeline_final, "public/model/model.pkl")

# Converte para JSON
print(json.dumps(result, indent=4))
