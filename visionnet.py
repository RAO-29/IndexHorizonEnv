from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.model import Model
from ray.rllib.models.misc import get_activation_fn, flatten
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class VisionNetwork(Model):
    """Generic vision network."""

    @override(Model)
    def _build_layers_v2(self, input_dict, num_outputs, options):
        inputs = input_dict["obs"]
        filters = options.get("conv_filters")
        if not filters:
            filters = _get_filter_config(inputs.shape.as_list()[1:])

        activation = get_activation_fn(options.get("conv_activation"))

        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = tf.layers.conv2d(
                    inputs,
                    out_size,
                    kernel,
                    stride,
                    activation=activation,
                    padding="same",
            data_format = "channels_last",
                    name="conv{}".format(i))
            out_size, kernel, stride = filters[-1]
            fc1 = tf.layers.conv2d(
                inputs,
                out_size,
                kernel,
                stride,
                activation=activation,
                padding="same",
        data_format = "channels_last",
                name="fc1")
            fc2 = tf.layers.conv2d(
                fc1,
                num_outputs, [1, 1],
                activation=None,
                padding="same",
        data_format = "channels_last",
                name="fc2")
            return flatten(fc2), flatten(fc1)


def _get_filter_config(shape):
    shape = list(shape)
    filters_84x84 = [
        [16, [8, 8], 4],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]
    filters_42x42 = [
        [16, [4, 4], 2],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]
    '''
    filters_8x61 = [
        [32, [8, 8], 4],
        [64, [4, 4], 2],
        [64, [3, 3], 1]
    ]
    
    filters_8x61 = [
        [50, [1, 5], 2],
        [100, [1, 5], 2],
        [200, [1, 5], 20]
    ]
'''
    if len(shape) == 3 and shape[:2] == [84, 84]:
        return filters_84x84
    elif len(shape) == 3 and shape[:2] == [42, 42]:
        return filters_42x42
    #elif len(shape) == 3 and shape[:2] == [8, 61]:
    #    return filters_8x61
    else:
        return filters_84x84
    
