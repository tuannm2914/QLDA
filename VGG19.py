from keras.models import Sequential
from keras.layers import Dense, Activation,MaxPooling2D,Conv2D,Flatten,Dropout
from keras.layers.normalization import BatchNormalization

def vgg():
    model = Sequential()
    VGG19_para =  [ 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    model.add(Conv2D(64, kernel_size=3, padding=1,input_shape=(48,48,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    for x in  VGG19_para:
        if x == 'M':
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        else:
            model.add(Conv2D(x,kernel_size = 3,padding= 1))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(7))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))
    return model



