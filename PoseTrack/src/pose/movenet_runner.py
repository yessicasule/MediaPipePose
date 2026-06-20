import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class MoveNetRunner:
    def __init__(self, model_name="movenet_lightning"):
        if model_name == "movenet_lightning":
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.input_size = 192
        else:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            self.input_size = 256
        self.model = module.signatures['serving_default']

    def process(self, image_rgb):
        input_image = tf.image.resize_with_pad(image_rgb, self.input_size, self.input_size)
        input_image = tf.cast(input_image, dtype=tf.int32)
        input_image = tf.expand_dims(input_image, axis=0)
        outputs = self.model(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores[0, 0]  # shape [17, 3]: [y, x, score]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
