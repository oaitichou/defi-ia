# -*- coding: utf-8 -*-
"""
@author: aaznague

"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
from keras.models import model_from_json
from google.colab import files

#Se connecter au Drive
drive.mount('/content/drive')

size = 128

y = np.load("/content/drive/My Drive/y_"+str(size)+".npy", 'r')
X = np.load("/content/drive/My Drive/X_"+str(size)+".npy", 'r')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=123)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=2)

print("X_train shape = {}".format(X_train.shape))
print("X_valid shape = {}".format(X_valid.shape))

#Génération d'images transformées
image_gen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=.2,
    height_shift_range=.2,
    horizontal_flip=True,
    vertical_flip=True)

image_gen.fit(X_train)

#Définition de modèle
batch_size = 256
epochs = 500
num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=(128,128,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(LeakyReLU(alpha=0.1))           
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer = RMSprop(),metrics = ['accuracy'])
model.summary()

history = model.fit_generator(image_gen.flow(X_train, y_train, batch_size=batch_size), verbose=1,
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(X_valid, y_valid))

scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Enregistrer le modèle
model_json = model.to_json()
with open("/content/drive/My Drive/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("/content/drive/My Drive/model.h5")


# Plus tard..
#Importation de modèle
json_file = open('/content/drive/My Drive/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/content/drive/My Drive/model.h5")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_valid, y_valid, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

data_test = np.load("/content/drive/My Drive/X_test_"+str(size)+".npy", 'r')
predictions = model.predict_classes(data_test)

predictions = pd.DataFrame(predictions, columns=['label']).to_csv('predictions.csv', sep = ';')
files.download("prediction.csv")
