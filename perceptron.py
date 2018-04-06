# This Python file uses the following encoding: utf-8
import random
# import matplotlib.pyplot as plt
# from scipy.interpolate import spline
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

def weight_init(num_inputs):
    """
    Função que inicializa os pesos e bias aleatoriamente utilizando numpy
    Parâmetro: num_inputs - quantidade de entradas X
    Retorna: w,b - pesos e bias da rede inicializados
    """
    w = [random.random() for x in range(num_inputs)]  #  w = None
    b = random.random()  #  b = None
    # print w, b
    return w,b

def activation_func(func_type, z):
    """
    Função que implementa as funções de ativação mais comuns
    Parãmetros: func_type - uma string que contém a função de ativação desejada
                z - vetor com os valores de entrada X multiplicado pelos pesos
    Retorna: saída da função de ativação
    """
    ### Seu código aqui (~2 linhas)
    if func_type == 'sigmoid':
        return None
    elif func_type == 'tanh':
        return None
    elif func_type == 'relu':
        return None
    elif func_type == 'degrau':
        return None
    elif func_type == 'signum':
        return -1 if z < 0 else 1 if z > 0 else 0

def forward(w,b,X):
    """
    Função que implementa a etapa forward propagate do neurônio
    Parâmetros: w - pesos
                b - bias
                X - entradas
    """
    z = sum(X[i]*w[i] for i in range(len(X))) + b  #  z = None
    out = activation_func('signum', z)  #  out = None
    # print z, out
    return out

def perceptron(X,y, num_interaction, learning_rate):
    """
    Função que implementa o loop do treinamento
    Parâmetros: x - entrada da rede
                y - rótulos/labels
                num_interaction - quantidade de interações desejada para a rede convergir
                learning_rate - taxa de aprendizado para cálculo do erro
    """
    #Passo 1 - Inicie os pesos e bias (~1 linha)
    w, b = weight_init(3)  #  w,b = None
    #Passo 2 - Loop por X interações
    for j in range(num_interaction):
        # Passo 3 -  calcule a saída do neurônio (~1 linha)
        y_pred = forward(w,b,X)  #  y_pred = None
        # Passo 4 - calcule o erro entre a saída obtida e a saída desejada nos rótulos/labels (~1 linha)
        erro = y - y_pred  #  erro = None
        # Passo 5 - Atualize o valor dos pesos (~1 linha)
        # Dica: você pode utilizar a função np.dot e a função transpose de numpy
        w = [w[i] + X[i] * erro * learning_rate for i in range(len(X))]  #  w = None
        b = b + 1 * erro * learning_rate
        # print w

    return y_pred, w, b
    # Verifique as saídas
    # print 'y:', y_pred, 'w:', w, 'b:', b

    #Métricas de Avaliação
    # y_pred = predict(y_pred)
    # print('Matriz de Confusão:')
    # print(confusion_matrix(y, y_pred))
    # print('F1 Score:')
    # print(classification_report(y, y_pred))


Padrao_t = [([0.1, 0.1, 0.1], [+1, -1, -1, -1, -1, -1, -1, -1]),
            ([0.1, 0.1, 1.1], [-1, +1, -1, -1, -1, -1, -1, -1]),
            ([0.1, 1.1, 0.1], [-1, -1, +1, -1, -1, -1, -1, -1]),
            ([0.1, 1.1, 1.1], [-1, -1, -1, +1, -1, -1, -1, -1]),
            ([1.1, 0.1, 0.1], [-1, -1, -1, -1, +1, -1, -1, -1]),
            ([1.1, 0.1, 1.1], [-1, -1, -1, -1, -1, +1, -1, -1]),
            ([1.1, 1.1, 0.1], [-1, -1, -1, -1, -1, -1, +1, -1]),
            ([1.1, 1.1, 1.1], [-1, -1, -1, -1, -1, -1, -1, +1]),
            ]

Padrao_v = [([0, 0, 0], [+1, -1, -1, -1, -1, -1, -1, -1]),
            ([0, 0, 1], [-1, +1, -1, -1, -1, -1, -1, -1]),
            ([0, 1, 0], [-1, -1, +1, -1, -1, -1, -1, -1]),
            ([0, 1, 1], [-1, -1, -1, +1, -1, -1, -1, -1]),
            ([1, 0, 0], [-1, -1, -1, -1, +1, -1, -1, -1]),
            ([1, 0, 1], [-1, -1, -1, -1, -1, +1, -1, -1]),
            ([1, 1, 0], [-1, -1, -1, -1, -1, -1, +1, -1]),
            ([1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, +1]),
            ]

pesos = []
num_interaction = 10
learning_rate = 0.2
for X, y in Padrao_t:
    for i in range(10):
        pesos.append(perceptron(X,y[i],num_interaction,learning_rate))

for X, y in Padrao_t:
    for i in range(10):
        print X, forward(X,y[0],num_interaction,learning_rate)[0], \
             perceptron(X,y[1],num_interaction,learning_rate)[0], \
             perceptron(X,y[2],num_interaction,learning_rate)[0], \
             perceptron(X,y[3],num_interaction,learning_rate)[0], \
             perceptron(X,y[4],num_interaction,learning_rate)[0], \
             perceptron(X,y[5],num_interaction,learning_rate)[0], \
             perceptron(X,y[6],num_interaction,learning_rate)[0], \
             perceptron(X,y[7],num_interaction,learning_rate)[0]
