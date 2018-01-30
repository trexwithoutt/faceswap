from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from .PixelShuffler import PixelShuffler ##Input: (N,C∗upscale_factor2,H,W); Output: (N,C,H∗upscale_factor,W∗upscale_factor)

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

IMAGE_SHAPE = (64, 64, 3)
ENCODER_DIM = 1024


def conv(filters):
    def block(x):
        # 5X5 0pad 2stride
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block


def upscale(filters): # #of filters
    def block(x):
        # 3X3 0pad
        x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block


def Encoder():
    input_ = Input(shape=IMAGE_SHAPE) #64x64x3
    x = input_
    x = conv(128)(x)
    x = conv(256)(x)
    x = conv(512)(x)
    x = conv(1024)(x)
    x = Dense(ENCODER_DIM)(Flatten()(x)) #encoder_dim = 1024
    x = Dense(4 * 4 * 1024)(x)
    x = Reshape((4, 4, 1024))(x) #reshape to 4x4x1024
    x = upscale(512)(x)
    return Model(input_, x)


def Decoder():
    input_ = Input(shape=(8, 8, 512))
    x = input_
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale(64)(x)
    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x)


encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

x = Input(shape=IMAGE_SHAPE)

autoencoder_A = Model(x, decoder_A(encoder(x)))
autoencoder_B = Model(x, decoder_B(encoder(x)))
autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error') #adam
autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')
