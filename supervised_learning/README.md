# Classifier

Este Algoritmo visa classificar se o câncer é benigno ou maligno.

## INFORMAÇÕES

Variável Alvo é variával que se pretende atingir.

Variável previsores são o conjunto de variáveis previsoras com varipaveis categóricas transformadas em numéricas, sem escalonamento.

Variável previsores_esc são o conjunto de variáveis previsoras com varipaveis categóricas transformadas em numéricas, com escalonamento.


[CATBOOST](https://catboost.ai/en/docs/) - 97.37% de acerto com os dados do teste; 98.83% de acerto com os dados do treino. 96.29% Validação Cruzada.

[LIGHTGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html) - 97.37% de acerto com os dados do teste; 98.83% de acerto com os dados do treino. 95.41% Validação Cruzada.

RANDOM FOREST - [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - 94.30% de acerto com os dados do teste; 95.89% de acerto com os dados do treino. 94.01% Validação Cruzada.

ÁRVORE DE DECISÃO - [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/tree.html) - 93.86% de acerto com os dados do teste; 97.36% de acerto com os dados do treino. 93.12% Validação Cruzada.

APRENDIZAGEM BASEADA EM INSTÂNCIAS (KNN)- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) - 96.05% de acerto com os dados do teste; 97.07% de acerto com os dados do treino. 96.27% Validação Cruzada.

[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - 98.68% de acerto com os dados do teste; 98.83% de acerto com os dados do treino. 97.56% Validação Cruzada.

[SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - 97.81% de acerto com os dados do teste; 98.24% de acerto com os dados do treino. 97.53% Validação Cruzada.

Naive Bayes - [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) - 90.79% de acerto com os dados do teste; 94.13% de acerto com os dados do treino. 93.12% Validação Cruzada.

Redes Neurais Artificiais - [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) - 97.37% de acerto com os dados do teste; 98.24% de acerto com os dados do treino. 98.06% Validação Cruzada. 




Link da tabela de dados utilizada - [DATASET](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)