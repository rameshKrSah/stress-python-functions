from tensorflow import keras


def encoder_block(input_shape):
    """
        Encoder block for the self-supervised learning task. Pass in the input shape.

        This block must be followed by dense layers for classification or regression on pretext
        tasks. After training on pretext tasks extract the encoder block freeze the weights and use
        it to learn the original problem with transfer learning.

        The first layer of the encoder is named 'encoder_input_layer' and the last layer is named
        'encoder_final_layer'.
    """
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv1D(filters=32, kernel_size=24, strides=1, padding='valid', activation=keras.activations.relu,
                            name='encoder_input_layer')(inputs)

    x = keras.layers.MaxPool1D(pool_size=4, strides=2)(x)

    x = keras.layers.Conv1D(filters=64, kernel_size=16, strides=1, activation=keras.activations.relu,
                            padding='valid')(x)

    x = keras.layers.MaxPool1D(pool_size=4, strides=2)(x)

    x = keras.layers.Conv1D(filters=96, kernel_size=8, strides=1, activation=keras.activations.relu,
                            padding='valid')(x)
    
    x = keras.layers.MaxPool1D(pool_size=4, strides=2)(x)
    
    x = keras.layers.Conv1D(filters=128, kernel_size=4, strides=1, padding='valid', activation=keras.activations.relu, 
                            name='encoder_final_layer')(x)

    return x


class PretextModel(keras.Model):
    """
        Create a TensorFlow model for a pretext task with inputs of shape (input_shape) and 
        number of output classes (n_classes)
    """
    def get_config(self):
        pass

    def __init__(self, input_shape, n_classes):
        super(PretextModel, self).__init__()
        self.in_shape = input_shape
        self.n_classes = n_classes

        # get the encoder block
        self.encoder_block = encoder_block(self.in_shape)

        # add some layers for pretext tasks
        self.max_pool =  keras.layers.MaxPool1D(pool_size=4, strides=2, name='max_pool')

        self.conv1 = keras.layers.Conv1D(filters=64, kernel_size=4, strides=1, name='conv1',
                                         activation=keras.activations.relu),
            
        self.gb_max_pool = keras.layers.GlobalMaxPooling1D(name='gb_max'),
            
        self.dense = keras.layers.Dense(units=512, activation=keras.activations.relu, name='dense_1'),
            
        self.out_layer = keras.layers.Dense(units=self.n_classes, activation=keras.activations.sigmoid,
                                            name='output_layer')

    def call(self, input):
        x = self.encoder_block(input)
        x = self.max_pool(x)
        x = self.conv1(x)
        x = self.gb_max_pool(x)
        x = self.dense(x)
        x = self.out_layer(x)

        return x


    def get_encoder_block(self):
        return self.encoder_block
