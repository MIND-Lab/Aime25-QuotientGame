import tensorflow as tf
import numpy as np
from scipy.interpolate import CubicSpline

class GradCam:
    def __init__(self, testset, model):
        self.testset = testset
        self.model = model


    def make_gradcam_heatmap(self, conv_layer_name):
        heatmaps = []
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(conv_layer_name).output, self.model.output]
        )
        for el in self.testset:
            el = np.expand_dims(el, 0)
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(el)
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, last_conv_layer_output)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output * pooled_grads
            heatmaps.append(heatmap.numpy())
        return np.array(heatmaps)

    def cubic_spline_interpolation(self, vector, v_size, input_size):
        heatmap = np.squeeze(vector)
        size_feature_map = len(heatmap.flatten())
        feature_map_flat = np.squeeze(heatmap.flatten())
        feature_map_flat = np.expand_dims(feature_map_flat, 1)

        x = np.linspace(0, size_feature_map - 1, v_size)
        f = CubicSpline(x, feature_map_flat, bc_type='natural')
        x_wanted = np.linspace(0, size_feature_map - 1, input_size)
        return_vector = f(x_wanted)

        return_vector = np.where(return_vector > 0, return_vector, 0)
        return return_vector

    def matrix_interpolation(self, vector):
        return np.mean(vector, axis = 1)

    def normalize_vector(self, vector):
        sum = vector.sum()
        return vector / sum