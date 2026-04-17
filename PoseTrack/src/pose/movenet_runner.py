import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class MoveNetRunner:
    def __init__(self, model_name="movenet_lightning"):
        # "movenet_lightning" or "movenet_thunder"
        if model_name == "movenet_lightning":
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.input_size = 192
        else:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            self.input_size = 256
        self.model = module.signatures['serving_default']

    def process(self, image_rgb):
        """Processes RGB image and returns landmarks matching MediaPipe indices (optional conversion) or standard 17 kpts"""
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        input_image = tf.image.resize_with_pad(image_rgb, self.input_size, self.input_size)
        input_image = tf.cast(input_image, dtype=tf.int32)
        input_image = tf.expand_dims(input_image, axis=0)

        outputs = self.model(input_image)
        # Sequence is [1, 1, 17, 3]
        keypoints_with_scores = outputs['output_0'].numpy()
        
        # We return the raw TF keypoints: [y, x, score] (normalized coords)
        # Let's map it roughly to something our system can understand or just return the standard TF output for now.
        return keypoints_with_scores[0, 0]

    def close(self):
        pass
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
