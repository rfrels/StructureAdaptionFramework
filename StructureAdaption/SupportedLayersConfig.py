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


import ml_collections
import tensorflow as tf
import tensorflow.keras.initializers as initializers
import StructureAdaption.utilities as utils

# Even more advanced approach: generate config from computation graph.
# Determine growing or pruning changes from computation graph.
# Also probably possible to generalize pruning from computational graph
# Not possible! There are multiple computational graphs for one @tf.function and the python level parameters wouldn't
# change


class AxisInitPair:
    """ If no weights or an initializer is provided this gives a default initializer for
    the respective weight. """
    def __init__(self, axis, default_initializer):
        self.axis = axis
        self.def_init = default_initializer


def get_config():
    """Builds and returns config.
    Represents supported layers.
    """
    config = ml_collections.ConfigDict()

    # config.Activation = ml_collections.ConfigDict()
    # config.Activation.layer_class = gp_layers.ScalingReg
    # config.Activation.tags = ['intermediate']
    # config.Activation.units_name = None
    # config.Activation.change_units = None
    # config.Activation.change_intermediate = {}
    # config.Activation.change_inputs = None

    config.BatchNormalization = ml_collections.ConfigDict()
    config.BatchNormalization.layer_class = tf.keras.layers.BatchNormalization
    config.BatchNormalization.tags = ['intermediate']
    config.BatchNormalization.units_name = None
    config.BatchNormalization.change_units = None
    config.BatchNormalization.change_intermediate = {'gamma:0': AxisInitPair(-1, initializers.Ones()),
                                                     'beta:0': AxisInitPair(-1, initializers.Zeros()),
                                                     'moving_mean:0': AxisInitPair(-1, initializers.Zeros()),
                                                     'moving_variance:0': AxisInitPair(-1, initializers.Zeros())}
    config.BatchNormalization.change_inputs = None

    config.Conv2D = ml_collections.ConfigDict()
    config.Conv2D.layer_class = tf.keras.layers.Conv2D
    config.Conv2D.tags = ['grow_prun_able', 'gradmax']
    config.Conv2D.units_name = 'filters'
    config.Conv2D.change_units = {'kernel:0': -1, 'bias:0': -1}
    config.Conv2D.change_intermediate = None
    config.Conv2D.change_inputs = {'kernel:0': -2}

    config.Dense = ml_collections.ConfigDict()
    config.Dense.layer_class = tf.keras.layers.Dense
    config.Dense.tags = ['grow_prun_able', 'gradmax']
    config.Dense.units_name = 'units'
    config.Dense.change_units = {'kernel:0': -1, 'bias:0': -1}
    config.Dense.change_intermediate = None
    config.Dense.change_inputs = {'kernel:0': -2}

    config.ScalingReg = ml_collections.ConfigDict()
    config.ScalingReg.layer_class = utils.ScalingReg
    config.ScalingReg.tags = ['intermediate']
    config.ScalingReg.units_name = None
    config.ScalingReg.change_units = None
    config.ScalingReg.change_intermediate = {'scaling_factor:0': AxisInitPair(-1, initializers.Ones())}
    config.ScalingReg.change_inputs = None

    return config
