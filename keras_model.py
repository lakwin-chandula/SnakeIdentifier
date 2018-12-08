from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
from keras.layers import BatchNormalization


# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'D:\\My Projects\\Dataset\\dataset4_grouped\\train'
validation_data_dir = 'D:\\My Projects\\Dataset\\dataset4_grouped\\validate'


nb_train_samples = 208 #75
nb_validation_samples = 125 #50
epochs = 5
batch_size = 4

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#start neural network
model = Sequential()

#convo 1st layer
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

#convo 2nd layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

#covo 3rd layer
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

#fully connected 1st layer
model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

#fully connected final layer
model.add(Dense(5))                  
#model.add(Activation('sigmoid'))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

keras_model = "my_keras_model.h5"
model.save('my_keras_model.h5')

#convert into tflite
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model)
tflite_model = converter.convert()
open("my_converted_model.tflite", "wb").write(tflite_model)
