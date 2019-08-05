from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Dropout, Flatten, Dense, Activation
from keras.layers.advanced_activations import ReLU
from keras.optimizers import Adam

def modelConstruct(sample_box=16):
    # print("model start")
    model = Sequential()
    model.add(Conv3D(2, (8, 8, 8), padding='same', input_shape=(sample_box, sample_box, sample_box, 4)))
    # keras.models.Sequential.output_shape
    model.add(ReLU())
    model.add(Conv3D(4, (8, 8, 8), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(8, (4, 4, 4), padding='same'))
    model.add(ReLU())
    # model.add(Convolution3D(96, 4, 4, 4))
    model.add(Conv3D(16, (4, 4, 4), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, (2, 2, 2), padding='same'))
    model.add(ReLU())
    model.add(Conv3D(64, (2, 2, 2), padding='same'))
    model.add(ReLU())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    opt = Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model