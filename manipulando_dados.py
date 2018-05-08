# This Python file uses the following encoding: utf-8
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split, KFold
from numpy import percentile
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Passo 1 - Leitura do dataset
dataset = read_csv('titanic3.csv') # Problemas com a biblioteca para leitura de xls
print(dataset.shape)

# Instâncias com informações faltando (NaN) para determinados atributos
dataset = dataset.dropna(axis=0, subset=['embarked']) # drop
print(dataset.shape)

print(dataset.describe())
dataset = dataset.fillna(dataset.mean()['age':'age']) # fill
dataset = dataset.fillna(dataset.mean()['fare':'fare']) # fill

# Dados discrepantes e outliers
dataset.boxplot()
P = percentile(dataset['fare'], [99])
dataset = dataset[(dataset['fare'] <= P[0])]
print(dataset.shape)

# Dados desbalanceados
print(dataset.groupby('survived').count())

# Passo 2 - Separar atributos e classes com redução da dimensão - name ticket	cabin	boat	body	home.dest
x = DataFrame(dataset, columns='pclass	sex	age	sibsp	parch	fare	embarked'.split('	'))
y = DataFrame(dataset['survived'])

x['sex'] = x['sex'].apply(lambda x: x.replace('female', '1').replace('male', '0')).astype('int')
x['embarked'] = x['embarked'].apply(lambda x: x.replace('C', '0').replace('Q', '0.5').replace('S', '1')).astype('float')

# Correlação dos atributos
print(x.corr())
scatter_matrix(x)
plt.show()

# Holdout
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

X = tf.placeholder(tf.float32, shape=[None,7], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None,1], name = 'Y')

w1 = tf.Variable(tf.random_uniform([7,7], -1, 1), name="w1")
w2 = tf.Variable(tf.random_uniform([7,1], -1, 1), name="w2")

b1 = tf.Variable(tf.zeros([7]), name="b1")
b2 = tf.Variable(tf.zeros([1]), name="b2")

hidden = tf.sigmoid(tf.add(tf.matmul(X, w1),b1)) #camada escondida
y_pred = tf.sigmoid(tf.add(tf.matmul(hidden,w2),b2)) #camada de saída

loss = tf.reduce_mean(tf.squared_difference(y_pred, Y))
# L2 Regularization with beta=0.01
regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
loss = tf.reduce_mean(loss + 0.01 * regularizers)

optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

train_losses = []
test_losses = []
init = tf.global_variables_initializer()
sess = tf.Session()

writer = tf.summary.FileWriter("./logs", sess.graph) #tensorboard

sess.run(init)

for epoch in range(100):
    sess.run(optimizer, feed_dict={X: X_train, Y: y_train})
    train_loss = sess.run(loss, feed_dict={X: X_train, Y: y_train})
    test_loss = sess.run(loss, feed_dict={X: X_test, Y: y_test})
    train_losses.append(train_loss)
    test_losses.append(test_loss)

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Testing loss')
plt.legend()
plt.ylim()
plt.show()

pred = []
for i in range(len(X_test)):
    y_ = round(sess.run(y_pred, feed_dict={X: [X_test.values[i]]})[0][0])
    pred.append(y_)

print('Resultados com Holdout:')
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print('---------------------------------')

# Validaçaão Cruzada
kf = KFold(n_splits=3)
print('Resultados com Validação Cruzada:')

accu_list = []
for k, (train_index, test_index) in enumerate(kf.split(X=x)):
    X_train, X_test = x.values[train_index], x.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]

    for epoch in range(80):
        sess.run(optimizer, feed_dict={X: X_train, Y: y_train})
        train_loss = sess.run(loss, feed_dict={X: X_train, Y: y_train})
        test_loss = sess.run(loss, feed_dict={X: X_test, Y: y_test})
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plt.plot(train_losses, label='{0}-Training loss'.format(k+1))
    plt.plot(test_losses, label='{0}-Testing loss'.format(k+1))
    plt.legend()
    plt.ylim()
    plt.show()

    pred = []
    for i in range(len(X_test)):
        y_ = round(sess.run(y_pred, feed_dict={X: [X_test[i]]})[0][0])
        pred.append(y_)

    accu = accuracy_score(y_test, pred)
    accu_list.append(accu)
    print(accu)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

print('Acuracia total: {0}'.format(sum(accu_list) / len(accu_list)))