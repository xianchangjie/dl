import tensorflow as tf

def net():
    sequential = tf.keras.models.Sequential()
    sequential.add(tf.keras.layers.Conv2D(filters=96,kernel_size=11,strides=4,activation='relu'))
    sequential.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
    sequential.add(tf.keras.layers.Conv2D(filters=256,kernel_size=5,padding='same',activation='relu'))
