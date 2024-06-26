import numpy as np
import pandas as pd

# Otimização de hiperparâmetros

df = pd.read_csv('path_Cancer', sep=',', encoding='iso-8859-1')

# Transformando as classes strings em variáveis categórica ordinais
df2 = pd.DataFrame.copy(df)

pd.set_option('future.no_silent_downcasting', True)

df2.replace({'diagnosis': {'M':1, 'B':0}}, inplace=True)


# Atributos previsores e alvo
alvo = df2.iloc[:, 1].values

previsores = df2.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]].values


# Escalonamento
from sklearn.preprocessing import StandardScaler

previsores_esc = StandardScaler().fit_transform(previsores)


# BASE DE TREINO E TESTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


alvo = label_encoder.fit_transform(alvo)


x_treino, x_teste, y_treino, y_teste = train_test_split(previsores_esc, alvo, test_size = 0.4, random_state = 0)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


############## CRIAÇÃO DO ALGORITMO ##############
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(random_state=0)

# Determinando o espaço do hiperparâmetro
param_grid = dict(
    n_estimators=[2, 50, 80, 100, 200, 300],
    learning_rate=[0.05, 0.1, 0.2, 0.5],
    max_depth=[1,2,3,4, 5],
    min_samples_split= [2, 5, 10],
    min_samples_leaf= [1, 2, 4],
    subsample= [0.8, 0.9, 1.0]
    )

# 96.05 - 100%
# {'learning_rate': 0.5, 
# 'max_depth': 4, 
# 'min_samples_leaf': 4, 
# 'min_samples_split': 2, 
# 'n_estimators': 100,
#  'subsample': 0.9}

# Configurando a procura com o Grid search
grid_search = GridSearchCV(gbm, param_grid, scoring='roc_auc', cv=4)
# Configurando os melhores hiperparâmetros
grid_search.fit(x_treino, y_treino)

print(grid_search.best_params_)

resultado = pd.DataFrame(grid_search.cv_results_)

# Ordenando os melhores resultados
resultado.sort_values(by='mean_test_score', ascending=False, inplace=True)

resultado.reset_index(drop=True, inplace=True)

print(resultado[['param_max_depth', 'param_learning_rate', 'param_n_estimators',
    'mean_test_score', 'std_test_score']].head())

# Melhor modelo encontrado pelo Grid Search
melhor_model = grid_search.best_estimator_

# Treinamento do modelo final com todo o conjunto de dados
melhor_model.fit(x_treino, y_treino)

test_score = melhor_model.score(x_teste, y_teste)
print(test_score)

# predicao = melhor_model.predict(x_teste)



previsoes = melhor_model.predict(x_teste)

print("\n\t........... Resultados .........\n")


################ Verification ####################
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Acurácia: %.2f%%" % (accuracy_score(y_teste, previsoes) * 100.00))

print('Matriz de correlação: \n', confusion_matrix(y_teste, previsoes))

print(classification_report(y_teste, previsoes))

print("\n\t........... Analisando os dados de treino (Overffitting) ...........\n")

previsoes_treino = melhor_model.predict(x_treino)

print("%.2f%%" % (accuracy_score(y_treino, previsoes_treino) * 100.00))

print("Matriz de correlação: \n", confusion_matrix(y_treino, previsoes_treino))


