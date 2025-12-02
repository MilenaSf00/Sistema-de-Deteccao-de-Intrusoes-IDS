import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

file_path = 'Wednesday-workingHours.pcap_ISCX.csv' 

try:
    print(f"1. Carregando dataset: {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)

    # --- LIMPEZA (Seu código original mantido e otimizado) ---
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    
    print("\n--- Contagem de Classes (Original) ---")
    print(df['Label'].value_counts())

    # --- PRÉ-PROCESSAMENTO PARA IA ---
    # Transforma tudo que não é 'BENIGN' em 'ATAQUE' (Classificação Binária)
    # Isso simplifica o problema para a apresentação e melhora os resultados
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    print("\n--- Contagem após binarização (0=Normal, 1=Ataque) ---")
    print(df['Label'].value_counts())

    # Amostragem (Opcional: Se seu PC for lento, descomente a linha abaixo para usar 10% dos dados)
    # df = df.sample(frac=0.1, random_state=42)

    X = df.drop('Label', axis=1)
    y = df['Label']

    # Codificar variáveis categóricas (Strings -> Números)
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])

    # Divisão Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- 1. MODELO BASELINE (TODAS AS FEATURES) ---
    print("\n🚀 Treinando Modelo Baseline (Todas as Features)...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    time_baseline = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    acc_baseline = accuracy_score(y_test, y_pred)
    print(f"Tempo Baseline: {time_baseline:.2f}s | Acurácia: {acc_baseline:.4f}")

    # --- 2. SELEÇÃO DE FEATURES (FEATURE IMPORTANCE) ---
    print("\n🔍 Selecionando Melhores Features...")
    # Pega as importâncias calculadas pelo Random Forest
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plota as Top 10 Features (SALVE ESSA IMAGEM PARA O SLIDE 5)
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Features Mais Importantes")
    plt.bar(range(10), importances[indices[:10]], align="center")
    plt.xticks(range(10), X.columns[indices[:10]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

    # Aplica a seleção (Pega apenas as features acima da média)
    sfm = SelectFromModel(rf, threshold='mean')
    sfm.fit(X_train, y_train)
    X_train_sel = sfm.transform(X_train)
    X_test_sel = sfm.transform(X_test)
    
    num_features_orig = X_train.shape[1]
    num_features_sel = X_train_sel.shape[1]
    print(f"Redução de Features: {num_features_orig} -> {num_features_sel}")

    # --- 3. MODELO OTIMIZADO (FEATURES SELECIONADAS) ---
    print("\n🚀 Treinando Modelo Otimizado...")
    start_time = time.time()
    rf_sel = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train_sel, y_train)
    time_opt = time.time() - start_time
    
    y_pred_sel = rf_sel.predict(X_test_sel)
    acc_opt = accuracy_score(y_test, y_pred_sel)
    print(f"Tempo Otimizado: {time_opt:.2f}s | Acurácia: {acc_opt:.4f}")

    # --- RESULTADOS FINAIS ---
    print("\n=== RELATÓRIO FINAL ===")
    print(f"Acurácia Original: {acc_baseline:.4f} vs Otimizada: {acc_opt:.4f}")
    print(f"Tempo de Treino: {time_baseline:.2f}s vs {time_opt:.2f}s")
    print(f"Melhora de velocidade: {time_baseline/time_opt:.1f}x mais rápido")

    # Matriz de Confusão (SALVE PARA O SLIDE 6)
    cm = confusion_matrix(y_test, y_pred_sel)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Ataque'], yticklabels=['Normal', 'Ataque'])
    plt.title('Matriz de Confusão (Modelo Otimizado)')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.savefig('confusion_matrix.png')
    plt.show()

except FileNotFoundError:
    print(f"ERRO: Baixe o arquivo 'Wednesday-workingHours.pcap_ISCX.csv' e coloque na pasta!")