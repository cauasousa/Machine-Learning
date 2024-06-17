# MLPClassifier

Este Algoritmo visa classificar se o câncer é benigno ou maligno.

## INFORMAÇÕES

Redes Neurais Artificiais - MLPClassifier - 97.37% de acerto com os dados do teste; 98.24% de acerto com os dados do treino. 98.06% Validação Cruzada. 

A quantidade de neurônios seguiu o seguinte cálculo:
    - QNT -> (Ne + Ns)/2 = (31 + 1) / 2 = 16 neurônios
                    OR
    - QNT ->  ((2 * Ne)/3) + Ns = ((2 * 31)/3) + 1 =  21 or 22 or 23

Variável Alvo é variával que se pretende atingir.

Variável previsores são o conjunto de variáveis previsoras com varipaveis categóricas transformadas em numéricas, sem escalonamento.

Variável previsores_esc são o conjunto de variáveis previsoras com varipaveis categóricas transformadas em numéricas, com escalonamento.

Para mais informações sobre o algoritmo MLPClassifier visite a [documentação](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

Link da tabela de dados utilizada - [DATASET](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)