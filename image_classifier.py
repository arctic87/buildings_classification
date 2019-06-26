# -*- coding: utf-8 -*- #кодировка utf для поддержки русских шрифтовзшз
import keras #машинное обучение
import numpy as np #математическая библиотека
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from PIL import ImageFile


# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'validation'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 1)
# Количество эпох
epochs = 6
# Размер мини-выборки
batch_size = 10
# Количество изображений для обучения
nb_train_samples = 60
# Количество изображений для проверки
nb_validation_samples = 30
# Количество изображений для тестирования
nb_test_samples = 15

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale') #перевод изображений в ч/б

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    color_mode = 'grayscale')

#Сверточная нейросеть
# Создаем последовательную модель
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(32, (3, 3),padding="same", input_shape=(img_width, img_height, 1)))
model.add(Activation('relu'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))

# Второй сверточный слой
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

# Сохраняем веса модели
model.save_weights('models/modelv2_CNN.h5')

# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("models/modelv2_CNN.json", "w")
json_file.write(model_json)
json_file.close()

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))



