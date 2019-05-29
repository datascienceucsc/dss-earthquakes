import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X_train = pd.read_csv('../../data/processed/X_train.csv', header = None)
y_train = pd.read_csv('../../data/processed/y_train.csv', header = None)
X_test = pd.read_csv('../../data/processed/X_test.csv', header = None)

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='sigmoid')) # defined hidden layer
model.add(Dense(1, activation='sigmoid')) #specifiy single node
model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics=['accuracy'])

h = model.fit(x=X_train, y=y_train, verbose = 1, batch_size = 20, epochs=100)
plt.plot(h.history['acc'])
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.title('accuracy')

prediction = model.predict(X_test)
#plt datapoints
