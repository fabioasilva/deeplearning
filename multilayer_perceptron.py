# This Python file uses the following encoding: utf-8
from __future__ import division
import random, math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from scipy.interpolate import spline
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score

def dot(X, w):
    """
    Função que soma o produto dos elementos de duas listas
    Parâmetro: X, w - entradas e pesos em formato de lista
    Retorna: soma dos produtos dos elementos de X por w
    """
    return sum(X[i]*w[i] for i in range(len(X)))

def weight_init(input_size, num_hidden, output_size):
    """
    Função que inicializa os pesos e bias aleatoriamente utilizando random
    Parâmetro: num_inputs - quantidade de entradas X
               num_hidden - quantidade de camadas ocultas
    Retorna: w,b - pesos e bias da rede inicializados
    """
    # inicializa os pesos e bias para os neuronios da camada oculta
    random.seed(0)

    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]

    # inicializa os pesos e bias para os neuronios da camada de saída
    output_layer = [[random.random() for __ in range((num_hidden or input_size) + 1)]
                    for __ in range(output_size)]

    return hidden_layer, output_layer

def activation_func(func_type, z):
    """
    Função que implementa as funções de ativação mais comuns
    Parãmetros: func_type - uma string que contém a função de ativação desejada
                z - vetor com os valores de entrada X multiplicado pelos pesos
    Retorna: saída da função de ativação
    """
    ### Seu código aqui (~2 linhas)
    if func_type == 'sigmoid':
        return 1 / (1 + math.exp(-z))
    elif func_type == 'tanh':
        return math.tanh(z)
    elif func_type == 'relu':
        return max(0, z)
    elif func_type == 'degrau':
        return 0 if z < 0 else 1 if z > 0 else 0.5
    elif func_type == 'signum':
        return -1 if z < 0 else 1 if z > 0 else 0
    elif func_type == 'linear':
        return z

def derivative_func(func_type, z):
    """
    Função que implementa as derivadas das funções de ativação mais comuns
    Parãmetros: func_type - uma string que contém a derivada da função de ativação desejada
                z - vetor com os valores de entrada X multiplicado pelos pesos
    Retorna: saída da derivada da função de ativação
    """
    ### Seu código aqui (~2 linhas)
    if func_type == 'sigmoid':
        return z * (1 - z)
    elif func_type == 'tanh':
        return (1 - z**2) / 2
    elif func_type == 'relu':
        return 1 if z >= 0 else 0
    elif func_type == 'degrau':
        return 0 if z <= 0 else 1
    elif func_type == 'signum':
        return 0 if z == 0 else 1
    elif func_type == 'linear':
        return 0 if z == 0 else 1

def forward(neural_network, X, activation_fn):
    """
    Função que implementa a etapa forward propagate da rede neural
    Parâmetros: neural_network - lista (camadas) de listas (neurônios) de listas (pesos com bias)
                X - entradas
    """
    outputs = []
    for n, layer in enumerate(neural_network):
        X_with_bias = X + [1]  #  para facilitar a multiplicação X * Ws, inclui o peso 1 para o bias
        # OR para o caso de não haver a camada oculta - perceptron de camada única
        layer_output = [activation_func(activation_fn if n == 1 else 'tanh', dot(X_with_bias, neuron)) for neuron in layer] or X
        X = layer_output

        outputs.append(layer_output)
    return outputs

def backpropagation(training_matrix, neural_network, learning_rate, activation_fn, num_epoch, batch_size, momentum=0):
    """
    Função que implementa o loop do treinamento com backpropagation
    Parâmetros: x - entrada da rede
                y - rótulos/labels
                neural_network - lista (camadas) de listas (neurônios) de listas (pesos com bias)
                num_interaction - quantidade de interações desejada para a rede convergir
                learning_rate - taxa de aprendizado para cálculo do erro
    """
    training_len = len(training_matrix)

    E_med_training = []

    old_out_deltas = []
    old_hid_deltas = []
    for __ in range(num_epoch):

        E_med = 0

        num_batch = 0

        for X, y in training_matrix:

            hidden_outputs, outputs = forward(neural_network, X, activation_fn)

            E_med += sum([(y[i] - output)**2 for i, output in enumerate(outputs)])

            num_batch += 1

            if num_batch == 1:
                # deltas da camada de saída
                output_deltas = [derivative_func(activation_fn, output) * (y[i] - output)
                                 for i, output in enumerate(outputs)]

                # deltas da camada oculta
                hidden_deltas = [derivative_func('tanh', hidden_output) *
                                 dot(output_deltas, [n[i] for n in neural_network[-1]])
                                 for i, hidden_output in enumerate(hidden_outputs)]
            else:
                output_deltas = [output_deltas[i] + delta
                                 for j, delta in enumerate([derivative_func(activation_fn, output) * (y[i] - output)
                                                            for i, output in enumerate(outputs)])]

                hidden_deltas = [hidden_deltas[i] + delta
                                 for j, delta in enumerate([derivative_func(activation_fn, hidden_output) *
                                                            dot(output_deltas, [n[i] for n in neural_network[-1]])
                                                            for i, hidden_output in enumerate(hidden_outputs)])]

            if num_batch in [batch_size, training_len]:
                # ajusta os pesos para a camada de saída
                for i, output_neuron in enumerate(neural_network[-1]):
                    for j, hidden_output in enumerate(hidden_outputs + [1]):
                        output_neuron[j] += learning_rate * output_deltas[i] * hidden_output
                        if momentum and old_out_deltas:
                            output_neuron[j] += momentum * old_out_deltas[i]

                old_out_deltas = output_deltas

                # ajusta os pesos para a camada oculta
                for i, hidden_neuron in enumerate(neural_network[0]):
                    for j, input in enumerate(X + [1]):
                        hidden_neuron[j] += learning_rate * hidden_deltas[i] * input
                        if momentum and old_hid_deltas:
                            hidden_neuron[j] += momentum * old_hid_deltas[i]

                old_hid_deltas = hidden_deltas

                num_batch = 0

        E_med_training.append(E_med / (2*training_len))

    return E_med_training

#-------------------------------------------------------------------------

# Questão 1
#
# paterns = [([0, 0, 0], [+1, -1, -1, -1, -1, -1, -1, -1]),
#            ([0, 0, 1], [-1, +1, -1, -1, -1, -1, -1, -1]),
#            ([0, 1, 0], [-1, -1, +1, -1, -1, -1, -1, -1]),
#            ([0, 1, 1], [-1, -1, -1, +1, -1, -1, -1, -1]),
#            ([1, 0, 0], [-1, -1, -1, -1, +1, -1, -1, -1]),
#            ([1, 0, 1], [-1, -1, -1, -1, -1, +1, -1, -1]),
#            ([1, 1, 0], [-1, -1, -1, -1, -1, -1, +1, -1]),
#            ([1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, +1]),
#            ]
#
# training_set = [([x[0]+random.uniform(-1,1)/10, x[1]+random.uniform(-1,1)/10, x[2]+random.uniform(-1,1)/10], y)
#                 for x, y in paterns
#                 for __ in range(10)]
#
# input_size = 3  # vertices do cubo
# num_hidden = 0  # não temos neurônios na camada oculta - Perceptron de Rosenblatt (Perceptron de camada única)
# output_size = 8  # oito neurônios na camada de saída, um para cada um dos oito padroes
#
# # inicializa os pesos e bias para os neuronios da camada de saída - Perceptron de Rosenblatt (Perceptron de camada única)
# hidden_layer, output_layer = weight_init(input_size, num_hidden, output_size)
# network = [hidden_layer, output_layer]
#
# learning_rate = 0.5
# activation_fn = 'signum'
# num_epoch = 15
# batch_size = 1
# momentum = 0.01
#
# print 'Conjunto de treinamento:', len(training_set)
# for x, y in training_set[7:13]:
#     print x, y
#
# erro = backpropagation(paterns, network, learning_rate, activation_fn, num_epoch, batch_size, momentum)
#
# print 'Erro quadratico medio:', erro
# x = [range(len(erro))]
# plt.scatter(x, erro)
# plt.show()
#
# print 'Conjunto de validaçao:'
# for input_vector, target_vector in paterns:
#     outputs = forward(network, input_vector, activation_fn)[-1]
#     print input_vector, target_vector, outputs
#

# Questão 3
#
# treina_xor = [([0, 0], [0]),
#               ([0, 1], [1]),
#               ([1, 0], [1]),
#               ([1, 1], [0]),
#               ]
#
# treina_sen = [([x],[math.sin(math.pi * x) / (math.pi * x)]) for x in [random.uniform(0,4) for _ in range(1000)]]
#
# input_size = 2  # entradas do XOR/sen
# num_hidden = 6 # XOR = 2
# output_size = 1
#
# # inicializa os pesos e bias para os neuronios da camada oculta e de saída
# hidden_layer, output_layer = weight_init(input_size, num_hidden, output_size)
# network = [hidden_layer, output_layer]
#
# # parametros XOR
# # learning_rate = 5
# # activation_fn = 'sigmoid'
# # num_epoch = 300
# # batch_size = 1
# # momentum = 0
#
# # parametros sen
# learning_rate = 0.5
# activation_fn = 'tanh'
# num_epoch = 150
# batch_size = 1
# momentum = 0
#
#
# print 'Conjunto de treinamento:', len(treina_sen)
# for x, y in treina_sen[:10]:
#     print x, y
#
# erro = backpropagation(treina_sen, network, learning_rate, activation_fn, num_epoch, batch_size, momentum)
#
# print 'Erro quadratico medio:', erro
# x = [range(len(erro))]
# plt.scatter(x, erro)
# plt.show()
#
# print 'Conjunto de validaçao:'
# # for input_vector, target_vector in treina_xor:
# #     outputs = forward(network, input_vector, activation_fn)[-1]
# #     print input_vector, target_vector, outputs
#
# x = []
# y = []
# z = []
# for input_vector, target_vector in treina_sen:
#     y_pred = forward(network, input_vector, activation_fn)[-1]
#     x.append(input_vector[0])
#     y.append(target_vector[0])
#     z.append('b')
#     x.append(input_vector[0])
#     y.append(y_pred[0])
#     z.append('g')
# plt.scatter(x,y,color=z,marker='.')
# plt.show()

#
# Questão 4
#
# padrao = [[x,y] for x, y in [(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(10000)] if x**2 + y**2 <= 1]
# rotulo = [[3,7,2,6,4,8,1,5][4*(x>0)+2*(y>0)+(abs(y)>1-abs(x))] for x,y in padrao]
# rotulo_byte = []
# for i in rotulo:
#     #rotulo_bit = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]][i-1]
#     rotulo_bit = [-1,-1,-1,-1,-1,-1,-1,-1]
#     rotulo_bit[i-1] = 1
#     rotulo_byte.append(rotulo_bit)
#
# padrao_rotulo = zip(padrao, rotulo_byte)
#
# input_size = 2  # coordenadas x,y
# num_hidden = 6  # 1 para cada padrão de reta
# output_size = 8  # one-of-c-classes
#
# # inicializa os pesos e bias para os neuronios da camada oculta e de saída
# hidden_layer, output_layer = weight_init(input_size, num_hidden, output_size)
# network = [hidden_layer, output_layer]
#
# learning_rate = 0.5
# activation_fn = 'signum'
# num_epoch = 100
# batch_size = 1
# momentum = 0.01
#
# print 'Conjunto de treinamento:', len(padrao_rotulo)
# for i in range(10):
#     print padrao[i], rotulo[i], rotulo_byte[i]
#
# cores = [['red','green','blue','gray','gray','blue','green','red'][4*(x>0)+2*(y>0)+(abs(y)>1-abs(x))] for x,y in padrao]
# x = [x for x, __ in padrao]
# y = [y for __, y in padrao]
# plt.scatter(x,y,color=cores,marker='.')
# plt.show()
#
# erro = backpropagation(padrao_rotulo, network, learning_rate, activation_fn, num_epoch,
#                        batch_size, momentum)
#
# print 'Erro quadratico medio:', erro
# x = [range(len(erro))]
# plt.scatter(x, erro)
# plt.show()
#
# padrao = [[x,y] for x, y in [(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(10000)] if x**2 + y**2 <= 1]
# rotulo = [[3,7,2,6,4,8,1,5][4*(x>0)+2*(y>0)+(abs(y)>1-abs(x))] for x,y in padrao]
#
# print 'Conjunto de validaçao:', len(padrao)
# y_pred = []
# for i in range(len(padrao)):
#     outputs = forward(network, padrao[i], activation_fn)[-1]
#     y_pred.append(outputs)
#     if i < 10:
#         print padrao[i], rotulo[i], outputs
#
# cores = [['green','blue','red','gray','red','gray','green','blue','black'][8 if sum(y) != -6 else dot([0,1,2,3,4,5,6,7], [0 if i < 0 else 1 for i in y])] for y in y_pred]
# #cores = [['green','blue','red','gray','red','gray','green','blue','black'][dot([4,2,1], [0 if i < 0 else 1 for i in y])] for y in y_pred]
# x = [x for x, __ in padrao]
# y = [y for __, y in padrao]
# plt.scatter(x,y,color=cores,marker='.')
# plt.show()
#
# y = [8 if sum(y) != -6 else dot([0,1,2,3,4,5,6,7], [0 if i < 0 else 1 for i in y]) for y in y_pred]
#
# print(confusion_matrix(rotulo, y))

#
# Questão 5
#
# ne = 10 # número de entradas anteriores
# np = 3 # número de passos posteriores
#
# function_set = [math.sin(x/10 + math.sin(x/10)**2) for x in range(0,120+np)]
#
# treina_predicao = [([function_set[x-i] for i in range(1, ne+1)], [function_set[x+j] for j in range(0, np)]) for x in range(ne, len(function_set)-np+1)]
#
# input_size = ne  # entradas do x(n) = sen(n + sen2(n))
# num_hidden = ne
# output_size = np
#
# # inicializa os pesos e bias para os neuronios da camada oculta e de saída
# hidden_layer, output_layer = weight_init(input_size, num_hidden, output_size)
# network = [hidden_layer, output_layer]
#
# learning_rate = 0.25
# activation_fn = 'tanh'
# num_epoch = 1000
# batch_size = 1
# momentum = 0
#
# print 'Conjunto de treinamento:', len(treina_predicao)
# for x, y in treina_predicao[:10]:
#     print x, y
#
# erro = backpropagation(treina_predicao, network, learning_rate, activation_fn, num_epoch,
#                        batch_size)  # , momentum, old_out_deltas, old_hid_deltas)
#
# print 'Erro quadratico medio:', erro
# x = [range(len(erro))]
# plt.scatter(x, erro)
# plt.show()
#
# print 'Conjunto de validaçao:'
# x = [x/10 for x in range(0, len(function_set))]
# y = function_set
# z= [0 for __ in x]
# for i in range(ne, len(function_set)-np+1):
#     input_vector = [function_set[i-j] for j in range(1, ne+1)]
#     y_pred = forward(network, input_vector, activation_fn)[-1]
#     for j in range(0, np):
#         x.append((i+j)/10)
#         y.append(y_pred[j])
#         z.append(j)
#
# plt.scatter(x,y,c=z,marker='.')
# plt.show()
