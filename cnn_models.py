from tensorflow.keras.layers import (Input, Reshape, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout, Dense,
                                     LeakyReLU, InputLayer)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

#for grad-CAM
def load_model_benchmark_functional(n_dims, number_classes):
    loss = 'categorical_crossentropy'
    metrics = ['categorical_accuracy']
    learning_rate = 0.00020441990333108206
    optimizer = Adam(learning_rate=learning_rate)
    initializer = HeUniform()

    input_layer = Input(shape=(n_dims,))
    x = Reshape((n_dims, 1))(input_layer)

    x = Conv1D(filters=100, kernel_size=100, strides=1, padding='same', activation='relu',
               kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.01)(x)

    x = Conv1D(filters=100, kernel_size=5, strides=2, padding='same', activation='relu',
               kernel_initializer=initializer)(x)
    x = MaxPooling1D(pool_size=6, strides=3, padding='same')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.01)(x)

    x = Conv1D(filters=25, kernel_size=9, strides=5, padding='same', activation='relu', kernel_initializer=initializer)(
        x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = Flatten()(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(units=732)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.7)(x)

    x = Dense(units=189)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.25)(x)

    x = Dense(units=152)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.1)(x)

    output_layer = Dense(units=number_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def load_model_benchmark(n_dims, number_classes):
    initializer = HeUniform()

    model = Sequential()
    model.add(InputLayer(shape=(n_dims,)))
    model.add(Reshape((n_dims, 1)))

    # ----- CNN layers
    model.add(Conv1D(filters=100,
                     kernel_size=100,
                     strides=1,
                     padding='same',
                     activation='relu',
                     kernel_initializer=initializer))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
    model.add(Conv1D(filters=100,
                     kernel_size=5,
                     strides=2,
                     padding='same',
                     activation='relu',
                     kernel_initializer=initializer))
    model.add(MaxPooling1D(pool_size=6,
                           strides=3,
                           padding='same'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
    model.add(Conv1D(filters=25,
                     kernel_size=9,
                     strides=5,
                     padding='same',
                     activation='relu',
                     kernel_initializer=initializer))
    model.add(MaxPooling1D(pool_size=3,
                           strides=2,
                           padding='same'))

    # ----- Flatten layer between CNN and Dense layers
    model.add(Flatten())
    model.add(Dropout(rate=0.1))
    # ----- Dense layers
    model.add(Dense(units=732))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.7))

    model.add(Dense(units=189))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.25))

    model.add(Dense(units=152))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.1))

    model.add(Dense(units=number_classes, activation='softmax'))
    return model