# Sistema de Detecção de Intrusões (IDS) com Seleção de Features

**Grupo 3:** Gabriel Bellagamba, Milena Ferreira, Luis Felipe Alves, Sabrina Fernandes, Danielly Cristina Neves

**Disciplina:** Engenharia de Sistemas de Detecção de Intrusões

**Dataset:**  [Link para dataset](https://storage.googleapis.com/kaggle-data-sets/3674161/6376134/compressed/Wednesday-workingHours.pcap_ISCX.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251202%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251202T181010Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2ddfe475d568289889215eae5ca121fe847e5b27cef42247d05e83beb55a1a11c6ed4adeda295b1b1c4b7614817e7e9ff0b2b8a6f15a792e76a541cf3ff922360c8b33e7d4e5cd824dfe4187d2213fa238b9678dd27c733053a1ae3cb73220fe4adaec78c5c14a9d831dff4c2fd02c6cb26482e39ea6dd104b68d9230ff572d8f76a35d921f03a08b9eb1dd1e49d40d7f65ff603c21611557bc2b326befe2711a48e11a4a3834fb4a60ce6c072cba86f69db4fbc6a3dfd6c21e251a46e56fdbb12e57c7b6b89213b643627aebedc435d4bfb58761017df7121a6a114909c074039b72c295d1831fc2c7438d544ce2316f0142b45808b38a17b14419562b42b63)

## 1. Objetivo

Este projeto visa desenvolver e otimizar um IDS baseado em IA, utilizando o algoritmo **Random Forest**. O foco principal é demonstrar como a **Seleção de Features** (Feature Selection) pode reduzir a complexidade computacional mantendo a alta precisão na detecção de ataques.

## 2. Metodologia

* **Dataset:** CIC-IDS2017 (Arquivo: `Wednesday-workingHours.pcap_ISCX.csv`).
  * Este subconjunto contém tráfego normal e ataques de **DoS/DDoS**, Heartbleed e Web Attacks.
* **Pré-processamento:** * Limpeza de dados (remoção de NaNs e Infinitos).
  * Codificação de variáveis categóricas (Label Encoding).
  * Binarização das classes (0 = Normal, 1 = Ataque).
* **Modelo de IA:** Random Forest Classifier.
* **Técnica de Seleção:** Embedded Method (Feature Importance).

## 3. Resultados Obtidos

O modelo foi treinado comparando o cenário com todas as features (78 colunas) versus o cenário otimizado.

### Principais Features Identificadas

O modelo identificou que as variáveis relacionadas ao **tamanho do pacote** são as mais críticas para identificar ataques de DoS neste dataset:

1. Max Packet Length
2. Packet Length Variance
3. Packet Length Mean

### Desempenho (Matriz de Confusão)

O modelo final obteve uma performance robusta:

* **Acurácia:** > 99%
* **Falsos Positivos:** Baixíssimo índice (apenas 24 em ~200k amostras de teste).
* **Falsos Negativos:** Baixíssimo índice (apenas 23 ataques não detectados).

![Confusion Matrix](./confusion_matrix.png)

## 4. Como Executar

1. Instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
