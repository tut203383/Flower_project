import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from keras.models import Model, Sequential
from keras import backend as K
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Input, Flatten
from sklearn.model_selection import KFold
from keras.applications import VGG16
from keras import regularizers

datagen = ImageDataGenerator(
        rescale=1./255,
        # rotation_range=30,
        # horizontal_flip=True,
        )

valid_datagen = ImageDataGenerator(rescale=1./255)

datagen_da = ImageDataGenerator(rescale=1./255, rotation_range=30, horizontal_flip=True)

img_width=128
img_height=128
bs=8

train_dir = '/content/drive/MyDrive/Flower_Project/Flowers/Train'
test_dir = '/content/drive/MyDrive/Flower_Project/Flowers/Test'

filenames_train = []
classes_train = []
filenames_test = []
classes_test = []

class_dirs_train = os.listdir(train_dir)
class_dirs_test = os.listdir(test_dir)
class_dirs_train = [x for x in class_dirs_train if x != '.DS_Store']

for class_dir in class_dirs_train:
    class_path = os.path.join(train_dir, class_dir)
    for filename in os.listdir(class_path):
        filepath = os.path.join(class_path, filename)
        filenames_train.append(filepath)
        classes_train.append(class_dir)

for class_dir in class_dirs_test:
    class_path = os.path.join(test_dir, class_dir)
    for filename in os.listdir(class_path):
        filepath = os.path.join(class_path, filename)
        filenames_test.append(filepath)
        classes_test.append(class_dir)

train_df = pd.DataFrame({
    "filename": filenames_train,
    "class": classes_train
})

test_df = pd.DataFrame({
    "filename": filenames_test,
    "class": classes_test
})

print(train_df)

train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                              x_col="filename", y_col="class", has_ext=True,
                                              class_mode="categorical", target_size=(img_width, img_height),
                                              batch_size=bs, color_mode='rgb')

validation_generator = valid_datagen.flow_from_dataframe(dataframe=test_df,
                                              x_col="filename", y_col="class", has_ext=True,
                                              class_mode="categorical", target_size=(img_width, img_height),
                                              batch_size=bs, color_mode='rgb')

train_generator_da = datagen_da.flow_from_dataframe(dataframe=train_df,
                                              x_col="filename", y_col="class", has_ext=True,
                                              class_mode="categorical", target_size=(img_width, img_height),
                                              batch_size=bs, color_mode='rgb')

# First model normal
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# H = model.fit(train_generator, epochs=300, verbose=2)
H = model.fit(train_generator, epochs=50, validation_data=validation_generator)
loss, accuracy = model.evaluate(validation_generator, verbose=0)

#model regularization
model_reg = Sequential()
model_reg.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model_reg.add(MaxPooling2D((2, 2)))
model_reg.add(Conv2D(64, (3, 3), activation='relu'))
model_reg.add(MaxPooling2D((2, 2)))
model_reg.add(Conv2D(128, (3, 3), activation='relu'))
model_reg.add(MaxPooling2D((2, 2)))
model_reg.add(Flatten())
model_reg.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model_reg.add(Dropout(0.5))
model_reg.add(Dense(10, activation='softmax'))

model_reg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# H_reg = model_reg.fit(train_generator, epochs=30, verbose=2)
H_reg = model_reg.fit(train_generator, epochs=50, validation_data=validation_generator)

# loss_reg, accuracy_reg = model_reg.evaluate(validation_generator, verbose=0)

#mdoel reg + da
model_da = Sequential()
model_da.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model_da.add(MaxPooling2D((2, 2)))
model_da.add(Conv2D(64, (3, 3), activation='relu'))
model_da.add(MaxPooling2D((2, 2)))
model_da.add(Conv2D(128, (3, 3), activation='relu'))
model_da.add(MaxPooling2D((2, 2)))
model_da.add(Flatten())
model_da.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model_da.add(Dropout(0.5))
model_da.add(Dense(10, activation='softmax'))

model_da.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# H_da = model_da.fit(train_generator_da, epochs=300, verbose=2)
H_da = model_da.fit(train_generator_da, epochs=50, validation_data=validation_generator)

model_f = Sequential()
model_f.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model_f.add(MaxPooling2D((2, 2)))
model_f.add(Conv2D(64, (3, 3), activation='relu'))
model_f.add(MaxPooling2D((2, 2)))
model_f.add(Conv2D(128, (3, 3), activation='relu'))
model_f.add(MaxPooling2D((2, 2)))
model_f.add(Flatten())
model_f.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model_f.add(Dropout(0.5))
model_f.add(Dense(10, activation='softmax'))

model_f.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fine tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

model_f = Sequential()
model_f.add(base_model)
model_f.add(Flatten())
model_f.add(Dense(512, activation='relu'))
model_f.add(Dropout(0.5))
model_f.add(Dense(10, activation='softmax'))

for layer in base_model.layers:
    layer.trainable = False

model_f.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# T = model_f.fit(train_generator, epochs=10, verbose=2)
H_f = model_f.fit(train_generator_da, epochs=50, validation_data=validation_generator)

# print out the evaluation 
loss, accuracy = model.evaluate(validation_generator, verbose=0)
print('First Test loss:', loss)
print('First Test accuracy:', accuracy)

loss_reg, accuracy_reg = model_reg.evaluate(validation_generator, verbose=0)
print('Reg Test loss:', loss_reg)
print('Reg Test accuracy:', accuracy_reg)

loss_da, accuracy_da = model_da.evaluate(validation_generator, verbose=0)
print('DA Test loss:', loss_da)
print('DA Test accuracy:', accuracy_da)

loss_f, accuracy_f = model_f.evaluate(validation_generator, verbose=0)
print('Test loss(Fine Tuning):', loss_f)
print('Test accuracy(Fine Tuning):', accuracy_f)

# plot the curve of training and validation 
plt.plot(H.history['accuracy'], color='tab:red', linestyle='--', label='CNN_train')
plt.plot(H.history['val_accuracy'], color='tab:red', linestyle='-', label='CNN_val')
plt.plot(H_reg.history['accuracy'], color='tab:blue', linestyle='--', label='CNN+Reg_train')
plt.plot(H_reg.history['val_accuracy'], color='tab:blue', linestyle='-', label='CNN+Reg_val')
plt.plot(H_da.history['accuracy'], color='tab:green', linestyle='-', label='CNN+Reg+DA_train')
plt.plot(H_da.history['val_accuracy'], color='tab:green', linestyle='--', label='CNN+Reg+DA_val')
plt.plot(H_f.history['accuracy'], color='tab:cyan', linestyle='--', label='CNN+Reg+DA+Fine_train')
plt.plot(H_f.history['val_accuracy'], color='tab:cyan', linestyle='-', label='CNN+Reg+DA+Fine_val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1, 0.5))
plt.show()

# Make predictions on the validation data
y_pred1 = np.argmax(model.predict(validation_generator), axis=-1)
y_true1 = validation_generator.classes

y_pred2 = np.argmax(model_reg.predict(validation_generator), axis=-1)
y_true2 = validation_generator.classes

y_pred3 = np.argmax(model_da.predict(validation_generator), axis=-1)
y_true3 = validation_generator.classes

y_pred4 = np.argmax(model_f.predict(validation_generator), axis=-1)
y_true4 = validation_generator.classes

cm = confusion_matrix(y_true1, y_pred1)
print(cm)

cm1 = confusion_matrix(y_true2, y_pred2)
print(cm1)

cm2 = confusion_matrix(y_true3, y_pred3)
print(cm2)

conf_matrix = confusion_matrix(y_true4, y_pred4)
print(conf_matrix)
