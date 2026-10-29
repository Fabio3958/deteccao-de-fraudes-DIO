# Detecção de Fraudes em Transações de Cartão de Crédito

Projeto do desafio de Machine Learning da **Digital Innovation One (DIO)**, dedicado à construção e avaliação de modelos preditivos para identificação de transações fraudulentas.

---

## 1. O problema:

Detectar fraudes em transações de cartão de crédito é um problema de **classificação binária**: cada transação é legítima (`Class = 0`) ou fraude (`Class = 1`). O desafio central não é o modelo em si, mas a **distribuição dos dados**: as fraudes representam cerca de **0,17%** do dataset (148 de 85.443 no conjunto de teste), enquanto as transações legítimas são a quase totalidade.

Esse desbalanceamento extremo **invalida a forma tradicional de avaliar modelos**:

- **Acurácia engana.** Todos os modelos avaliados marcaram `accuracy = 1.00` — e isso não diz praticamente nada. Um modelo que prevê "nunca é fraude" também acertaria ~99,8%, e seria inútil.
- **O custo dos erros é assimétrico.** Deixar uma fraude passar (falso negativo) gera prejuízo financeiro direto; um alarme falso (falso positivo) gera atrito, mas custo muito menor.
- **A métrica precisa focar na classe minoritária.** Por isso o projeto avalia os modelos com **recall, precision e F1 da classe de fraude (1)**, além de **AUC-ROC** e **curva Precision-Recall**.

Em resumo: em dados desbalanceados, **acertar a maioria é fácil; o valor está em detectar a minoria.**

---

## 2. Preparação dos dados

O dataset foi carregado diretamente de uma URL pública ([creditcard.csv](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)) com Pandas.

**Feature Engineering**
- Criação de `Amount_log` (`np.log1p`) para comprimir valores extremos e reduzir a assimetria da variável de valor da transação;
- Padronização de `Amount` com `StandardScaler`.

**Separação e divisão**
- Separação entre features (`X`) e alvo (`y = Class`);
- Divisão **estratificada** em treino (70%) e teste (30%) com `train_test_split(stratify=y, random_state=42)` preservando a proporção de fraudes. O teste ficou com **85.295 transações normais e 148 fraudes**. Sem estratificação, o teste poderia conter pouquíssimas (ou nenhuma) fraude, invalidando a avaliação.

**Tratamento do desbalanceamento — três estratégias comparadas**
1. **Undersampling:** manter todas as fraudes e sortear igual número de normais;
2. **SMOTE:** geração de fraudes sintéticas por interpolação;
3. **Pesos de classe:** `class_weight="balanced"` (Random Forest) e `scale_pos_weight=10` (XGBoost), sem reamostrar.

---

## 3. Comparação entre os modelos

Avaliação no **conjunto de teste original (desbalanceado)**, com foco na **classe de fraude (1)**:

| Modelo | Estratégia de balanceamento | Precision (fraude) | Recall (fraude) | F1 (fraude) |
|---|---|---|---|---|
| Regressão Logística | threshold ajustado (0,3) | **0,84** | 0,66 | 0,73 |
| Random Forest | `class_weight="balanced"` | **0,84** | 0,76 | 0,79 |
| Random Forest (Pipeline) | `class_weight="balanced"` | 0,79 | 0,71 | 0,75 |
| **XGBoost** | `scale_pos_weight=10` | 0,94 | **0,78** | **0,85** |

**Leitura dos resultados:**

- **Acurácia não diferencia nada:** todos os modelos marcam `accuracy = 1.00`. A diferença real está na **classe de fraude**.
- **Regressão Logística:** precision 0,84, recall 0,66 → detecta ~2/3 das fraudes, deixando passar ~1/3. F1 de 0,73.
- **Random Forest:** mesma precision (0,84) da Logística, mas **recall 0,76** — detecta mais fraudes sem aumentar falsos positivos. F1 sobe para 0,79.
- **Random Forest no Pipeline:** precision 0,79 e recall 0,71 (F1 0,75) — **ligeiramente abaixo** do RF "avulso". A diferença vem de que o Pipeline foi treinado nos dados **originais** sem as colunas derivadas `Amount_log`/`Amount_scaled` e com o scaler ajustado **apenas no treino** (sem leakage), o que é o comportamento **mais correto** — mas custa alguns pontos de desempenho aparente. Vale destacar que o RF avulso havia sido treinado com features extras e scaler ajustado em todos os dados, o que infla levemente as métricas.
- **XGBoost:** o grande vencedor — **precision 0,94** (quase todos os alertas são fraudes reais) e **recall 0,78** (detecta a maior parte das fraudes). **F1 = 0,85**, o melhor de todos. O boosting capturou padrões que os modelos anteriores não conseguiram.

**Conclusão da comparação:** o **XGBoost foi o melhor modelo**, com o maior F1 (0,85) e a melhor precision (0,94), mantendo recall superior ao Random Forest. Em detecção de fraude, esse equilíbrio — muitos alertas corretos e poucas fraudes escapando — é exatamente o que se busca.

---

## 4. Limiar de decisão e interpretabilidade com SHAP

### Limiar de decisão (threshold)

Por padrão, classificadores usam **0,5** como limiar. Mas, em fraude, esse limiar é **conservador demais** — ele otimiza um equilíbrio entre classes que não reflete o negócio. O projeto adotou um **limiar de 0,3**:

```python
threshold = 0.3
y_pred_custom = (y_probs > threshold).astype(int)
