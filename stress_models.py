import tensorflow as tf
from tensorflow import keras

def get_simple_cnn_model(input_shape, metrics, learning_rate):
    """Create a simple CNN model for EDA based stress classification. 

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
        keras.layers.Dense(units = 2, activation = tf.nn.softmax)
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, 
                       optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
                      metrics = metrics)
    
    return temp_model


def get_supervised_balance_model(input_shape, metrics, learning_rate):
    """After hyperparameterization optimization the best set of hyperparameters were:
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
        keras.layers.Dense(units = 2, activation = tf.nn.softmax)
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, 
                       optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
                      metrics = metrics)
    
    return temp_model

def get_supervised_full_model(input_shape, metrics, learning_rate):
    """After hyperparameterization optimization the best set of hyperparameters were:
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
                            input_shape = input_shape, padding='same'),
      
        keras.layers.Conv1D(filters = 100, kernel_size = (5), strides = 1, 
                            activation = tf.nn.relu, padding='same'),
        keras.layers.GlobalMaxPool1D(),
        
        keras.layers.Dense(units = 256, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.1),

        keras.layers.Dense(units = 128, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.1),
        
        keras.layers.Dense(units = 64, activation=tf.nn.relu),
        keras.layers.Dense(units = 2, activation = tf.nn.softmax)
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, 
                       optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
                      metrics = metrics)
    
    return temp_model