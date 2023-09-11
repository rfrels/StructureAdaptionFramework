#    StructureAdaptionFramework: a framework for handling neuron-level and layer-level structure adaptions in
#    neural networks.
#
#    Copyright (C) 2023  Roman Frels, roman.frels@gmail.com
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, version 3 of the License.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import tensorflow as tf
from keras import backend


class ScalingReg(tf.keras.layers.Layer):
    """ This layer introduces scaling_factors that are applied to the output of other layers
    for use in regularization Pruning.

    """
    initializer = None
    regularizer = None
    constraint = None
    instance_count = 0

    def __init__(self, **kwargs):
        kwargs.update(dict(name=f'scaling_reg_{ScalingReg.instance_count}'))
        ScalingReg.instance_count += 1
        super(ScalingReg, self).__init__(**kwargs)
        self.scaling_factors = None
        self.built = False

    def build(self, input_shape):
        n_neurons = input_shape[-1]
        self.scaling_factors = self.add_weight("scaling_factor", shape=[n_neurons],
                                               regularizer=self.regularizer,
                                               initializer=self.initializer,
                                               constraint=self.constraint,
                                               trainable=True)
        self.built = True

    def call(self, inputs):
        return tf.math.multiply(inputs, self.scaling_factors)

    def get_scaling_factors(self):
        return self.scaling_factors

    def get_config(self):
        config = super(ScalingReg, self).get_config()
        return config


class PolarizationRegularizer(tf.keras.regularizers.Regularizer):
    """A polarization regularizer that applies polarization regularisation
    according to: Neuron-level Structured Pruning using Polarization Regularizer,
    https://proceedings.neurips.cc/paper/2020/hash/703957b6dd9e3a7980e040bee50ded65-Abstract.html

    Arguments:
        l: Float; balancing factor that balances this regularization with other terms in the loss function.
        gamma: Float; Portion of channels that should maximally be pruned.
            Should have a small margin to the actually pruned channels.
            (In the paper this would be: 1 - rho)
    """

    def __init__(self, l=0.0001, gamma=0.5):
        if (gamma > 1 or gamma < 0):
            raise ValueError("gamma must be in the range [0, 1].")
        self.l = backend.cast_to_floatx(l)
        self.gamma = backend.cast_to_floatx(gamma)
        t = -2 + 4 * gamma
        self.t = backend.cast_to_floatx(t)

    def __call__(self, x):
        return self.l * (self.t * tf.reduce_sum(tf.abs(x))
                         - tf.reduce_sum(tf.abs(tf.subtract(x, tf.reduce_mean(x)))))

    def get_config(self):
        return {'l': float(self.l), 'gamma': float(self.gamma)}

    @classmethod
    def from_config(cls, config):
        # Standard implementation
        return cls(**config)


class RangeConstraint(tf.keras.constraints.Constraint):
    """ Range constraint, clipping values to range [min, max]."""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, w):
        return tf.clip_by_value(w, self.min, self.max)

    def get_config(self):
        return {
            'min_value': self.min,
            'max_value': self.max
        }
