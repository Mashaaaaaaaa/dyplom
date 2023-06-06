# імпорт необхідних бібліотек
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# розмір зображення 48х48 пікселів
size_of_picture = 48

# шлях до зображень
path = "../input/face-expression-recognition-dataset/images/"
# вивести приклади вхідних даних
expression = 'fear'

plt.figure(figsize= (12,12))
for i in range(1, 10, 1):
    plt.subplot(3,3,i)
    img = load_img(path+"train/"+expression+"/"+
                  os.listdir(path + "train/" + expression)[i], target_size=(size_of_picture, size_of_picture))
    plt.imshow(img)   
plt.show()
# вивести кількість зображень у наборі даних для навчання у кожному класі
for expression in os.listdir(path + "train"):
    print(str(len(os.listdir(path + "train/" + expression))) + " " + expression + " зображень")
# кількість ітерацій
batch_size = 128

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(path + "train",
                                                    target_size=(size_of_picture,size_of_picture),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(path + "validation",
                                                    target_size=(size_of_picture,size_of_picture),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
# кількість класів
classes = 7

# ініціалізація ЗНМ
model = Sequential()

# перший згортковий шар 
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# другий згортковий шар 
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# третій згортковий шар 
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# четвертий згортковий шар 
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# згладжування
model.add(Flatten())

# перший повноз'єднаний шар 
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
# другий повноз'єднаний шар
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(classes, activation='softmax'))

print(model.summary())
# кількість епох
epochs = 30

checkpoint = ModelCheckpoint("model1.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks_list = [early_stopping,checkpoint,reduce_learningrate]
#компіляція моделі
model.compile(loss='categorical_crossentropy',optimizer = Adam(learning_rate=0.0001),metrics=['accuracy'])
%%time
history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=train_generator.n//train_generator.batch_size,
                                epochs=epochs,
                                validation_data = validation_generator,
                                validation_steps = validation_generator.n//validation_generator.batch_size,
                                callbacks=callbacks_list)
# збереження навченої моделі
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model1.h5")
#побудова графіків втрат і точності
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.ylabel('Втрати', fontsize=16)
plt.plot(history.history['loss'], label='Навчальні втрати')
plt.plot(history.history['val_loss'], label='Валідаційні втрати')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Точність', fontsize=16)
plt.plot(history.history['accuracy'], label='Навчальна точність')
plt.plot(history.history['val_accuracy'], label='Валідаційна точність')
plt.legend(loc='lower right')
plt.show()
# обчислення передбачення
predictions = model.predict_generator(generator=validation_generator)
y_pred = [np.argmax(probas) for probas in predictions]
y_test = validation_generator.classes
class_names = validation_generator.class_indices.keys()

def plot_confusion_matrix(cm, classes, title='Матриця невідповідностей', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Вірні мітки')
    plt.xlabel('Передбачені мітки')
    plt.tight_layout()
    
# обчислення матриці невідповідностей
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# побудова матриці невідповідностей
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Матриця невідповідностей')
plt.show()
