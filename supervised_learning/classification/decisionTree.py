import numpy as np
import pandas as pd

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
from sklearn.tree import DecisionTreeClassifier

arvore = DecisionTreeClassifier(
        criterion='log_loss',
        random_state = 0, 
        max_depth=3,
        )


arvore.fit(x_treino, y_treino)
            

previsoes = arvore.predict(x_teste)

print("\n\t........... Resultados .........\n")


################ Verification ####################
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Acurácia: %.2f%%" % (accuracy_score(y_teste, previsoes) * 100.00))

print('Matriz de correlação: \n', confusion_matrix(y_teste, previsoes))

print(classification_report(y_teste, previsoes))

print("\n\t........... Analisando os dados de treino (Overffitting) ...........\n")

previsoes_treino = arvore.predict(x_treino)

print("%.2f%%" % (accuracy_score(y_treino, previsoes_treino) * 100.00))

print("Matriz de correlação: \n", confusion_matrix(y_treino, previsoes_treino))

print("\n\t........... Validação Cruzada ...........\n")

# Validação Cruzada
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits = 20, shuffle=True, random_state = 3)

modelo = DecisionTreeClassifier(criterion='log_loss', random_state = 0, max_depth=3)

resultado = cross_val_score(modelo, previsores_esc, alvo, cv = kfold)
print("Acurácia Média: %.2f%%" % (resultado.mean() * 100.0))