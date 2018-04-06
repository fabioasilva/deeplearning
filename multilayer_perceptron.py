from __future__ import division
# This Python file uses the following encoding: utf-8
import random, math
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from scipy.interpolate import spline
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

def dot(X, w):
    """
    Função que soma o produto dos elementos de duas listas
    Parâmetro: X, w - entradas e pesos em formato de lista
    Retorna: soma dos produtos dos elementos de X por w
    """
    return sum(X[i]*w[i] for i in range(len(X)))

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
        return None
    elif func_type == 'relu':
        return max(0, z)
    elif func_type == 'degrau':
        return None
    elif func_type == 'signum':
        return -1 if z < 0 else 1 if z > 0 else 0

def forward(neural_network, X):
    """
    Função que implementa a etapa forward propagate da rede neural
    Parâmetros: neural_network - lista (camadas) de listas (neurônios) de listas (pesos com bias)
                X - entradas
    """
    outputs = []
    for layer in neural_network:
        X_with_bias = X + [1]  #  para facilitar a multiplicação X * Ws, inclui o peso 1 para o bias
        layer_output = [activation_func('sigmoid', dot(X_with_bias, neuron)) for neuron in layer]
        X = layer_output

        outputs.append(layer_output)

    return outputs

def backpropagation(X,y, neural_network, learning_rate, momentum=0, old_out_deltas=[], old_hid_deltas=[]):
    """
    Função que implementa o loop do treinamento com backpropagation
    Parâmetros: x - entrada da rede
                y - rótulos/labels
                neural_network - lista (camadas) de listas (neurônios) de listas (pesos com bias)
                num_interaction - quantidade de interações desejada para a rede convergir
                learning_rate - taxa de aprendizado para cálculo do erro
    """
    hidden_outputs, outputs = forward(neural_network, X)

    # deltas da camada de saída
    output_deltas = [output * (1 - output) * (y[i] - output)
                     for i, output in enumerate(outputs)]

    # ajusta os pesos para a camada de saída
    for i, output_neuron in enumerate(neural_network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] += learning_rate * output_deltas[i] * hidden_output
            if momentum and old_out_deltas:
                output_neuron[j] += momentum * old_out_deltas[i]

    for i, delta in enumerate(output_deltas):
        try:
            old_out_deltas[i] = delta
        except:
            old_out_deltas.append(delta)

    # retropropagação dos erros para a camada oculta
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                      dot(output_deltas, [n[i] for n in neural_network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # ajusta os pesos para a camada oculta
    for i, hidden_neuron in enumerate(neural_network[0]):
        for j, input in enumerate(X + [1]):
            hidden_neuron[j] += learning_rate * hidden_deltas[i] * input
            if momentum and old_hid_deltas:
                hidden_neuron[j] += momentum * old_hid_deltas[i]

    for i, delta in enumerate(hidden_deltas):
        try:
            old_hid_deltas[i] = delta
        except:
            old_hid_deltas.append(delta)

#-------------------------------------------------------------------------

random.seed(0)
input_size = 2
num_hidden = 2 # 4 questão 4
output_size = 1

# inicializa os pesos e bias para os neuronios da camada oculta
hidden_layer = [[random.random() for __ in range(input_size + 1)]
                for __ in range(num_hidden)]

# inicializa os pesos e bias para os neuronios da camada de saída
output_layer = [[random.random() for __ in range(num_hidden + 1)]
                for __ in range(output_size)]

network = [hidden_layer, output_layer]

# Questão 2

treina_xor = [([0, 0], [0]),
              ([0, 1], [1]),
              ([1, 0], [1]),
              ([1, 1], [0]),
              ]

treina_sen = [([x],[math.sin(math.pi * x) / (math.pi * x)]) for x in [random.uniform(0,4) for _ in range(1000)]]

# pesos iniciais
print network

momentum = -0.4
old_out_deltas = []
old_hid_deltas = []
for __ in range(1000):
    for input_vector, target_vector in treina_sen:
        backpropagation(input_vector, target_vector, network, 0.25) # , momentum, old_out_deltas, old_hid_deltas)

print old_hid_deltas, old_out_deltas
# pesos finais
print network

cores = []
for input_vector, target_vector in treina_sen:
    outputs = forward(network, input_vector)[-1]
    cores.append(['r','g'][round(target_vector[0],2) == round(outputs[0],2)])
    print input_vector, target_vector, outputs

x = [x for x, __ in treina_sen]
y = [y for __, y in treina_sen]
plt.scatter(x,y,color=cores)
plt.show()

# Questão 4
#
# padrao = [[x,y] for x, y in [(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(1000)] if x**2 + y**2 <= 1]
# rotulo = [[[.3,.7,.2,.6,.4,.8,.1,.5][4*(x>0)+2*(y>0)+(abs(y)>1-abs(x))]] for x,y in padrao]
#
# print 'Pesos iniciais:', network
# for __ in range(1000):
#     for i in range(len(padrao)):
#         backpropagation(padrao[i], rotulo[i], network, 1.0)
#
# print 'Pesos finais', network
# print 'TESTE:'
# cores = [['red','green','blue','gray','gray','blue','green','red'][4*(x>0)+2*(y>0)+(abs(y)>1-abs(x))] for x,y in padrao]
# x = [x for x, __ in padrao]
# y = [y for __, y in padrao]
# plt.scatter(x,y,color=cores)
# plt.show()
#
# cores = []
# for i in range(len(padrao)):
#     outputs = forward(network, padrao[i])[-1]
#     cores.append(['r','g'][rotulo[i][0] == round(outputs[0],1)])
#     print padrao[i], rotulo[i], round(outputs[0],1), cores[i]
#
# x = [x for x, __ in padrao]
# y = [y for __, y in padrao]
# plt.scatter(x,y,color=cores)
# plt.show()
#
# padrao = [[x,y] for x, y in [(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(1000)] if x**2 + y**2 <= 1]
# rotulo = [[[.3,.7,.2,.6,.4,.8,.1,.5][4*(x>0)+2*(y>0)+(abs(y)>1-abs(x))]] for x,y in padrao]
#
# print 'VALIDACAO:'
# for i in range(len(padrao)):
#     outputs = forward(network, padrao[i])[-1]
#     print padrao[i], rotulo[i], round(outputs[0],1)
#
# cores = [['red','green','blue','gray','gray','blue','green','red'][4*(x>0)+2*(y>0)+(abs(y)>1-abs(x))] for x,y in padrao]
# x = [x for x, __ in padrao]
# y = [y for __, y in padrao]
# plt.scatter(x,y,color=cores)
# plt.show()

# Questão 5
#
# padrao = [([x/10],[math.sin(x/10 + math.sin(x/10)**2)]) for x in range(0,10)]
#
# print 'Pesos iniciais:', network
# for __ in range(1000):
#     for input_vector, target_vector in padrao:
#         backpropagation(input_vector, target_vector, network, 0.2)
#
# print 'Pesos finais', network
# for input_vector, target_vector in padrao:
#     outputs = forward(network, input_vector)[-1]
#     print input_vector, target_vector, outputs
#
# padrao = [([x/10],[math.sin(x/10 + math.sin(x/10)**2)]) for x in range(10,13)]
#
# for input_vector, target_vector in padrao:
#     outputs = forward(network, input_vector)[-1]
#     print input_vector, target_vector, outputs
#
# padrao = [(x/10,math.sin(x/10 + math.sin(x/10)**2)) for x in range(0,100)]
# x = [x for x, __ in padrao]
# y = [y for __, y in padrao]
# plt.scatter(x,y)
# plt.show()