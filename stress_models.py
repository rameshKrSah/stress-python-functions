from doctest import OutputChecker
from re import L
import tensorflow as tf
from tensorflow import keras


def get_cnn_batch_norm_model(input_shape, n_classes, learning_rate, metrics, prev_model=None):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu', name='conv1_relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu', name='conv2_relu')(conv2)

    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu', name='conv3_relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(n_classes, activation='softmax', name='output')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    if prev_model != None:
        #set the weights of the pre trained model
        for i in range(len(model.layers) - 1):
            model.layers[i].set_weights(prev_model.layers[i].get_weights())

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)
    return model


def get_simple_cnn_model(input_shape, metrics, learning_rate):
    """Create a simple CNN model for EDA based stress binary classification. 

    input_shape -- Tuple, needed for the first layer
    metrics -- list, metrics to optimize
    learning_rate -- float, learning rate for the Adam optimizer
    """

    temp_model = keras.Sequential([
        keras.layers.Conv1D(filters=100, kernel_size = (10), strides = 1, activation = tf.nn.relu, 
                            input_shape = input_shape, padding='same'),
      
        keras.layers.Conv1D(filters = 100, kernel_size = (5), strides = 1, 
                            activation = tf.nn.relu, padding='same'),
        keras.layers.GlobalMaxPool1D(),
        
        keras.layers.Dense(units = 264, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.3),

        keras.layers.Dense(units = 128, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.3),
        
        keras.layers.Dense(units = 64, activation=tf.nn.relu),
        keras.layers.Dense(units = 1, activation = tf.nn.sigmoid)
    ])
    
    temp_model.compile(loss = keras.losses.BinaryCrossentropy(), 
                       optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
                      metrics = metrics)
    
    return temp_model


def get_supervised_balance_adarp_model(input_shape, metrics, learning_rate):
    """Returns a CNN model for balanced stress dataset.
    After hyperparameterization optimization the best set of hyperparameters were:
        1. batch_size = 147
        2. CNN1 filters = 100
        3. CNN1 kernel size = 5
        4. CNN2 filters = 50
        5. CNN2 kernel size = 10
        6. Dense1 units = 128
        7. Dense2 units = 256
        8. Dense3 units = 64
        9. Dropout1 = 0.1
        10. Dropout2 = 0.3
        11. Learning rate = 0.002558
        12. Optimizer = Adam

        Returns the CNN model.
    """
    temp_model = keras.Sequential([
        keras.layers.Conv1D(filters=100, kernel_size = (5), strides = 1, activation = tf.nn.relu, 
                            input_shape = input_shape, padding='same'),
      
        keras.layers.Conv1D(filters = 50, kernel_size = (10), strides = 1, 
                            activation = tf.nn.relu, padding='same'),
        keras.layers.GlobalMaxPool1D(),
        
        keras.layers.Dense(units = 128, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.1),

        keras.layers.Dense(units = 256, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.3),
        
        keras.layers.Dense(units = 64, activation=tf.nn.relu),
        keras.layers.Dense(units = 1, activation = tf.nn.sigmoid)
    ])
    
    temp_model.compile(loss = keras.losses.BinaryCrossentropy(), 
                       optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
                      metrics = metrics)
    
    return temp_model

def get_supervised_full_adarp_model(input_shape, metrics, learning_rate):
    """Returns a large CNN model for binary stress classification.
    After hyperparameterization optimization the best set of hyperparameters were:
        1. Batch size = 100
        2. CNN1 filters = 250
        3. CNN1 kernel size = 5
        4. CNN2 filters = 100
        5. CNN2 kernel size = 5
        6. Dense1 units = 256
        7. Dense2 units = 128
        8. Dense3 units = 64
        9. Dropout1 = 0.1
        10. Dropout2 = 0.1
        11. Learning rate = 0.01157
        12. Optimizer = Adam
        
        Return the CNN model
    """
    temp_model = keras.Sequential([
        keras.layers.Conv1D(filters=250, kernel_size = (5), strides = 1, activation = tf.nn.relu, 
                            input_shape = input_shape, padding='same', name='conv1'),
      
        keras.layers.Conv1D(filters = 100, kernel_size = (5), strides = 1, 
                            activation = tf.nn.relu, padding='same', name='conv2'),
        keras.layers.GlobalMaxPool1D(),
        
        keras.layers.Dense(units = 256, activation = tf.nn.relu, name='dense1'),
        keras.layers.Dropout(rate = 0.1),

        keras.layers.Dense(units = 128, activation = tf.nn.relu, name='dense2'),
        keras.layers.Dropout(rate = 0.1),
        
        keras.layers.Dense(units = 64, activation=tf.nn.relu, name='dense3'),
        keras.layers.Dense(units = 2, activation = tf.nn.softmax, name='output-dense')
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, 
                       optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
                      metrics = metrics)
    
    return temp_model


def get_transfer_wesad_model(input_shape, metrics, learning_rate):
    """After hyperparameterization optimization the best set of hyperparameters were:
        1. Batch size = 120
        2. CNN1 filters = 100
        3. CNN1 kernel size = 10
        4. CNN2 filters = 50
        5. CNN2 kernel size = 5
        6. Dense1 units = 128
        7. Dense2 units = 64
        8. Dense3 units = 128
        9. Dropout1 = 0.2
        10. Dropout2 = 0.3
        11. Learning rate = 0.0154
        12. Optimizer = Adam
        
        For test loss of 0.2275
        Return the CNN model
    """
    temp_model = keras.Sequential([
        keras.layers.Conv1D(filters = 100, kernel_size = (5), strides = 1, activation = tf.nn.relu, 
                            input_shape = input_shape),
        
        keras.layers.Conv1D(filters = 100, kernel_size = (10), strides = 1, activation = tf.nn.relu),
        keras.layers.GlobalMaxPool1D(),
        
        keras.layers.Dense(units = 128, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.3),
        
        keras.layers.Dense(units = 64, activation = tf.nn.relu, name='penultimate_layer'),
        keras.layers.Dropout(rate = 0.2),
        keras.layers.Dense(units = 2, activation = tf.nn.softmax)
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, 
                       optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
                      metrics = metrics)
    
    return temp_model


def get_transfer_adarp_model(input_shape, metrics, learning_rate):
    """After hyperparameterization optimization the best set of hyperparameters were:
        1. Batch size = 106
        2. CNN1 filters = 250
        3. CNN1 kernel size = 2
        4. CNN2 filters = 100
        5. CNN2 kernel size = 10
        6. Dense1 units = 64
        7. Dense2 units = 128
        8. Dense3 units = 128
        9. Dropout1 = 0.1
        10. Dropout2 = 0.1
        11. Learning rate = 0.002046
        12. Optimizer = Adam
        
        For test loss of 0.5263
        Return the CNN model
    """
    temp_model = keras.Sequential([
        keras.layers.Conv1D(filters=250, kernel_size=(2), strides=1, activation=tf.nn.relu, 
                            input_shape=input_shape, padding='same'),
      
        keras.layers.Conv1D(filters=100, kernel_size=(10), strides=1, 
                            activation=tf.nn.relu, padding='same'),
        keras.layers.GlobalMaxPool1D(),
        
        keras.layers.Dense(units=64, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.1),

        keras.layers.Dense(units=128, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.1),
        
        keras.layers.Dense(units=128, activation=tf.nn.relu),
        keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
    ])
    
    temp_model.compile(loss=keras.losses.BinaryCrossentropy(), 
                       optimizer=keras.optimizers.Adam(learning_rate = learning_rate), 
                      metrics=metrics)
    
    return temp_model