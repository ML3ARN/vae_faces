import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Layer, Flatten, Dense, Reshape
                                        
class DownConvBlock(Layer):
    count = 0
    def __init__(self, filters, kernel_size=(3,3), strides=1, padding='same'):
        super(DownConvBlock, self).__init__(name=f"DownConvBlock_{DownConvBlock.count}")
        DownConvBlock.count+=1
        self.forward = Sequential([Conv2D(filters, kernel_size, strides, padding),
                                    BatchNormalization(), LeakyReLU(0.2)], name='deconv_seq')
        def call(self, x):
            return self.forward(x)


class UpConvBlock(Layer):
    count = 0
    def __init__(self, filters, kernel_size=(3,3), padding='same'):
        super(UpConvBlock, self).__init__(name=f"UpConvBlock_{UpConvBlock.count}")
        UpConvBlock.count+=1
        self.forward = Sequential([Conv2D(filters, kernel_size, 1, padding), 
        LeakyReLU(0.2), 
        UpSampling2D((2,2))], name='upconv_seq')

    def call(self, x):
        return self.forward(x)

class GaussianSampling(Layer):
    ''' this custom layer is needed for the reparametrization trick '''
    def call(self, x):
        means, logvar = x
        epsilon = tf.random.normal(shape=tf.shape(means), mean=0., stddev=1.)
        samples = means + tf.exp(0.5*logvar)*epsilon
        return samples

class Encoder(Layer):
    def __init__(self, z_dim, name='encoder'):
        super(Encoder, self).__init__(name=name)
        self.features_extract= Sequential([
            DownConvBlock(filters = 32, kernel_size=(3,3), strides=2),
            DownConvBlock(filters = 32, kernel_size=(3,3), strides=2),
            DownConvBlock(filters = 64, kernel_size=(3,3), strides=2),
            DownConvBlock(filters = 64, kernel_size=(3,3), strides=2),
            Flatten()
        ], name='encoder_seq')

        self.dense_mean =  Dense(z_dim, name='mean')
        self.dense_logvar = Dense(z_dim, name='logvar')

        self.sampler = GaussianSampling()

    def call(self, inputs):
        x = self.features_extract(inputs)
        mean = self.dense_mean(x)
        logvar = self.dense_logvar(x)
        z = self.sampler([mean, logvar])
        return z, mean, logvar

class Decoder(Layer):
    def __init__(self, z_dim, name='decoder'):
        super(Decoder, self).__init__(name=name)

        self.forward = Sequential([
            Dense(7*7*64, activation='relu'),
            Reshape((7,7,64)),
            UpConvBlock(filters=64, kernel_size=(3,3)),
            UpConvBlock(filters=64, kernel_size=(3,3)),
            UpConvBlock(filters=32, kernel_size=(3,3)),
            UpConvBlock(filters=32, kernel_size=(3,3)),
            Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same', activation='sigmoid')
        ], name='decoder_seq')

    def call(self, x):
        return self.forward(x)
