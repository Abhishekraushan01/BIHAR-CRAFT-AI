import tensorflow_hub as hub
import tensorflow as tf
from utils import load_img, imshow

# Load arbitrary image stylization model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

content_path = "../data/content/pattern.jpg"
style_path = "../data/style/craft_style.jpg"

content_image = load_img(content_path)
style_image = load_img(style_path)

stylized = model(tf.constant(content_image), tf.constant(style_image))[0]

tf.keras.preprocessing.image.save_img(
    "../outputs/style_transfer/stylized.jpg",
    stylized[0]
)

print("Style transfer done!")
imshow(stylized)
