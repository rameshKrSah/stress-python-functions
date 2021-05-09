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
        keras.layers.Dense(units = n_classes, activation = tf.nn.softmax)
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, 
                       optimizer = keras.optimizers.Adam(learning_rate = learning_rate), 
                      metrics = metrics)
    
    return temp_model