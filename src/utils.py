import tensorflow as tf
import numpy as np

def load_img(path, max_dim=512):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (max_dim, max_dim))
    return img[tf.newaxis, :]

def imshow(img):
    import matplotlib.pyplot as plt
    plt.imshow(np.squeeze(img))
    plt.axis('off')
