# This Python file uses the following encoding: utf-8
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam

def one_hot(true_labels):
    """
    Função que implementa o one-hot encoding
    Entrada: true_labels - array original com os labels
    Retorna: labels - conversão one-hot
    """
    return np_utils.to_categorical(true_labels)

### leitura do dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
### Visualizar instâncias
plt.imshow(x_train[5], cmap=plt.get_cmap('gray'))
plt.title(y_train[5])
plt.show()
### Print - informações das instâncias
img_rows, img_cols = x_train.shape[1], x_train.shape[2]  # dimensões da imagem

x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)

print('Instâncias de treinamento:', x_train.shape[0])
print('Instâncias de teste:', x_test.shape[0])
### Normalizar
x_train = x_train / 255.0
x_test =  x_test / 255.0

### Conversão do array de predições Y - classes
y_train = one_hot(y_train)
y_test = one_hot(y_test)

num_classes = y_train.shape[1]
print('classes:', num_classes)

### Definição de Topologia da Rede
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(img_rows * img_cols,)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

### Definir otimizador, função custo e modo do treinamento
batch_size = 1000
epochs = 10

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    # verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)

model.summary()

print('Test loss:', score[0])
print('Test accuracy:', score[1])