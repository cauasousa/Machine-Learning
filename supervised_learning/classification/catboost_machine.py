import numpy as np
import pandas as pd

df = pd.read_csv('path_Cancer', sep=',', encoding='iso-8859-1')

# Transformando as classes strings em variáveis categórica ordinais
df2 = pd.DataFrame.copy(df)

pd.set_option('future.no_silent_downcasting', True)

# Atributos previsores e alvo
alvo = df2.iloc[:, 1]

previsores = df2.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]


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
from catboost import CatBoostClassifier


catboost = CatBoostClassifier(task_type='CPU', 
                              iterations = 49, 
                              learning_rate=0.1, 
                              depth = 2, 
                              random_state = 5,
                              eval_metric="Accuracy",
                              )

catboost.fit( x_treino, y_treino, 
             verbose=False,
             )

previsoes = catboost.predict(x_teste)

print("\n\t........... Resultados .........\n")


################ Verification ####################
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Acurácia: %.2f%%" % (accuracy_score(y_teste, previsoes) * 100.00))

print('Matriz de correlação: \n', confusion_matrix(y_teste, previsoes))

print(classification_report(y_teste, previsoes))

print("\n\t........... Analisando os dados de treino (Overffitting) ...........\n")

previsoes_treino = catboost.predict(x_treino)

print("%.2f%%" % (accuracy_score(y_treino, previsoes_treino) * 100.00))

print("Matriz de correlação: \n", confusion_matrix(y_treino, previsoes_treino))

print("\n\t........... Validação Cruzada ...........\n")

# Validação Cruzada
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits = 20, shuffle=True, random_state = 5)

modelo = CatBoostClassifier(task_type='CPU', 
                              iterations = 49, 
                              learning_rate=0.1, 
                              depth = 2, 
                              random_state = 5,
                              eval_metric="Accuracy",
                              verbose=False
                              )

resultado = cross_val_score(modelo, previsores_esc, alvo, cv = kfold)
print("Acurácia Média: %.2f%%" % (resultado.mean() * 100.0))