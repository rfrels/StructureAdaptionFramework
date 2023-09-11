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

"""Tests for structure_adaption.py"""

import logging

import absl.testing.parameterized as parameterized
import StructureAdaption.structure_adaption as structure_adaption
import tensorflow as tf
import StructureAdaption.SupportedLayersConfig
import StructureAdaption.utilities as utils
import StructureAdaption.test_networks as test_networks


supp_lay_conf = StructureAdaption.SupportedLayersConfig.get_config()
GP_ModelParser = structure_adaption.GrowingPruningModelParser(supported_layers_config=supp_lay_conf)


def get_layer_adjust_config(layer):
    cls_name = type(layer).__name__
    return getattr(supp_lay_conf, cls_name)


class LayerTest(parameterized.TestCase, tf.test.TestCase):
    """ Tests adding and removing weights for the change kinds: units, intermediate and inputs. """
    @parameterized.named_parameters(
        ('dense', tf.keras.layers.Dense(3), (3, 4)),
        ('batchnorm', tf.keras.layers.BatchNormalization(), (2, 4)),
        ('conv2d', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4)),
        ('scaling_reg', utils.ScalingReg(), (2, 4))
    )
    def test_consistency(self, layer, input_shape):
        x = tf.random.uniform(input_shape)
        original_out = layer(x)
        wrapped_layer = GP_ModelParser.parse_layer(layer)
        new_out = wrapped_layer(x)
        self.assertAllClose(original_out, new_out)

    @parameterized.named_parameters(
        # highest input dimension is batch size
        ('dense', tf.keras.layers.Dense(3), (3, 4), 1),
        ('dense_5neuron', tf.keras.layers.Dense(3), (3, 4), 5),
        ('conv2d', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4), 1),
        ('conv2d_5neuron', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4), 5),
    )
    def test_add_units_zeros(self, layer, input_shape, n_new):
        x = tf.random.uniform(input_shape)
        original_out = layer(x)
        wrapped_layer = GP_ModelParser.parse_layer(layer)
        to_init_out = wrapped_layer(x)  # Layer needs to be called to have weights
        old_weights, old_biases = layer.get_weights()
        original_output_shape = original_out.get_shape()
        n_neurons_old = original_output_shape[-1]
        init_values = [tf.keras.initializers.Zeros(), tf.keras.initializers.Zeros()]
        wrapped_layer = structure_adaption.add(wrapped_layer, n_new, init_values, change_kind='units')
        new_out = wrapped_layer(x)
        # Check the output has the expected shape
        new_shape = original_output_shape[:-1] + [n_neurons_old + n_new]
        self.assertAllEqual(new_shape, new_out.get_shape())
        # Check the old neurons create same output
        self.assertAllClose(original_out, new_out[Ellipsis, :n_neurons_old])
        # Check the new neurons create zero output
        self.assertEqual(0, tf.math.count_nonzero(new_out[Ellipsis, n_neurons_old:]))
        new_weights, new_biases = wrapped_layer.get_weights()
        # Check the new weights are zero
        added_weights = new_weights[Ellipsis, n_neurons_old:]
        self.assertAllEqual(added_weights, tf.zeros_like(added_weights))
        # Check the new biases are zero
        added_biases = new_biases[n_neurons_old:]
        self.assertAllEqual(added_biases, tf.zeros_like(added_biases))
        # Check the old weights are same
        kept_weights = new_weights[Ellipsis, :n_neurons_old]
        self.assertAllEqual(old_weights, kept_weights)
        # Check the old biases are same
        kept_biases = new_biases[Ellipsis, :n_neurons_old]
        self.assertAllEqual(old_biases, kept_biases)

    @parameterized.named_parameters(
        ('batch_norm', tf.keras.layers.BatchNormalization(), (3, 4), 1),
        ('batch_norm_5neuron', tf.keras.layers.BatchNormalization(), (3, 5, 5, 4), 5),
    )
    def test_add_intermediate_zeros(self, layer, input_shape, n_new):
        x = tf.random.uniform(input_shape)
        original_out = layer(x)
        wrapped_layer = GP_ModelParser.parse_layer(layer)
        print('weights: ' + str(layer.weights))
        to_init_out = wrapped_layer(x)  # Layer needs to be called to have weights
        # New input after growing would have more features
        new_input_shape = input_shape[:-1] + (n_new,)
        new_x = tf.concat([x, tf.random.uniform(new_input_shape)], axis=-1)

        original_output_shape = original_out.get_shape()
        n_neurons_old = original_output_shape[-1]
        old_weights = layer.get_weights()
        init_values = [tf.keras.initializers.Zeros(), tf.keras.initializers.Zeros(),
                       tf.keras.initializers.Zeros(), tf.keras.initializers.Zeros()]
        wrapped_layer = structure_adaption.add(wrapped_layer, n_new, init_values, 'intermediate')
        new_out = wrapped_layer(new_x)
        # Check the output has the expected shape
        new_shape = original_output_shape[:-1] + [n_neurons_old + n_new]
        self.assertAllEqual(new_shape, new_out.get_shape())
        # Check the old neurons create same output
        self.assertAllClose(original_out, new_out[Ellipsis, :n_neurons_old])
        # Check the new neurons create zero output
        self.assertEqual(0, tf.math.count_nonzero(new_out[Ellipsis, n_neurons_old:]))
        new_weights = wrapped_layer.get_weights()
        # Check the new weights are zero
        for new_weight in new_weights:
            added_weight = new_weight[Ellipsis, n_neurons_old:]
            self.assertAllEqual(added_weight, tf.zeros_like(added_weight))
        # Check the old weights are same
        for new_weight, old_weight in zip(new_weights, old_weights):
            kept_weight = new_weight[Ellipsis, :n_neurons_old]
            self.assertAllEqual(old_weight, kept_weight)

    @parameterized.named_parameters(
        ('dense', tf.keras.layers.Dense(3), (3, 4), 1),
        ('dense_5neuron', tf.keras.layers.Dense(3), (3, 4), 5),
        ('conv2d', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4), 1),
        ('conv2d_5neuron', tf.keras.layers.Conv2D(3, 3), (3, 5, 5, 4), 5),
    )
    def test_add_inputs_zeros(self, layer, input_shape, n_new):
        x = tf.random.uniform(input_shape)
        original_out = layer(x)
        wrapped_layer = GP_ModelParser.parse_layer(layer)
        to_init_out = wrapped_layer(x)  # Layer needs to be called to have weights
        n_features = input_shape[-1]
        # New input after growing would have more features
        new_input_shape = input_shape[:-1] + (n_new,)
        new_x = tf.concat([x, tf.random.uniform(new_input_shape)], axis=-1)
        old_weights, old_biases = layer.get_weights()
        init_values = [tf.keras.initializers.Zeros(), tf.keras.initializers.Zeros()]
        wrapped_layer = structure_adaption.add(wrapped_layer, n_new, init_values, 'inputs')
        new_out = wrapped_layer(new_x)
        new_weights, new_biases = wrapped_layer.get_weights()
        # Output of the layer shouldn't change.
        self.assertAllClose(original_out, new_out)
        # Check biases are unchanged
        self.assertAllEqual(old_biases, new_biases)
        # Check the new weights are zero
        added_weights = new_weights[Ellipsis, n_features:, :]
        self.assertAllEqual(added_weights, tf.zeros_like(added_weights))
        # Check the old weights are same
        kept_weights = new_weights[Ellipsis, :n_features, :]
        self.assertAllEqual(old_weights, kept_weights)

    @parameterized.named_parameters(
        # highest input dimension is batch size
        ('dense', tf.keras.layers.Dense(10), (3, 4), [0, 2, 4, 6, 8]),
        ('conv2d', tf.keras.layers.Conv2D(10, 3), (3, 5, 5, 4), [0, 2, 4, 6, 8]),
    )
    def test_remove_units(self, layer, input_shape, prun_indcs):
        x = tf.random.uniform(input_shape)
        original_out = layer(x)
        n_prun = len(prun_indcs)
        wrapped_layer = GP_ModelParser.parse_layer(layer)
        to_init_out = wrapped_layer(x)  # Layer needs to be called to have weights
        old_weights, old_biases = layer.get_weights()
        original_output_shape = original_out.get_shape()
        n_neurons_old = original_output_shape[-1]
        wrapped_layer = structure_adaption.remove(wrapped_layer, prun_indcs, 'units')
        new_out = wrapped_layer(x)
        # Check the output has the expected shape
        new_shape = original_output_shape[:-1] + [n_neurons_old - n_prun]
        self.assertAllEqual(new_shape, new_out.get_shape())
        # Check the remaining neurons create same output as the corresponding old neurons
        self.assertAllClose(original_out[Ellipsis, 1:n_neurons_old:2], new_out)
        new_weights, new_biases = wrapped_layer.get_weights()
        # Check the remaining weights are same
        self.assertAllEqual(old_weights[Ellipsis, 1:n_neurons_old:2], new_weights)
        # Check the remaining biases are same
        self.assertAllEqual(old_biases[Ellipsis, 1:n_neurons_old:2], new_biases)

    @parameterized.named_parameters(
        ('scaling_factors', utils.ScalingReg(), (3, 10), [0, 2, 4, 6, 8]),
        ('scaling_factors_4indim', utils.ScalingReg(), (3, 5, 5, 10), [0, 2, 4, 6, 8]),
        ('batch_norm', tf.keras.layers.BatchNormalization(), (3, 10), [0, 2, 4, 6, 8]),
        ('batch_norm_4indim', tf.keras.layers.BatchNormalization(), (3, 5, 5, 10), [0, 2, 4, 6, 8]),
    )
    def test_remove_intermediate(self, layer, input_shape, prun_indcs):
        x = tf.random.uniform(input_shape)
        original_out = layer(x)
        n_prun = len(prun_indcs)
        wrapped_layer = GP_ModelParser.parse_layer(layer)
        to_init_out = wrapped_layer(x)  # Layer needs to be called to have weights
        new_x = x[Ellipsis, 1:10:2]
        original_output_shape = original_out.get_shape()
        n_neurons_old = original_output_shape[-1]
        old_weights = layer.get_weights()
        wrapped_layer = structure_adaption.remove(wrapped_layer, prun_indcs, 'intermediate')
        new_out = wrapped_layer(new_x)
        # Check the output has the expected shape
        new_shape = original_output_shape[:-1] + [n_neurons_old - n_prun]
        self.assertAllEqual(new_shape, new_out.get_shape())
        # Check the old neurons create same output
        self.assertAllClose(original_out[Ellipsis, 1:10:2], new_out)
        new_weights = wrapped_layer.get_weights()
        # Check the old weights are same
        for new_weight, old_weight in zip(new_weights, old_weights):
            kept_weight = old_weight[Ellipsis, 1:10:2]
            self.assertAllEqual(new_weight, kept_weight)

    @parameterized.named_parameters(
        # highest input dimension is batch size
        ('dense', tf.keras.layers.Dense(4), (3, 10), [0, 2, 4, 6, 8]),
        ('conv2d', tf.keras.layers.Conv2D(4, 3), (3, 5, 5, 10), [0, 2, 4, 6, 8]),
    )
    def test_remove_inputs(self, layer, input_shape, prun_indcs):
        x = tf.random.uniform(input_shape)
        original_out = layer(x)
        n_prun = len(prun_indcs)
        n_features = input_shape[-1]
        wrapped_layer = GP_ModelParser.parse_layer(layer)
        to_init_out = wrapped_layer(x)  # Layer needs to be called to have weights
        old_weights, old_biases = layer.get_weights()
        wrapped_layer = structure_adaption.remove(wrapped_layer, prun_indcs, 'inputs')
        new_x = x[Ellipsis, 1:10:2]
        new_out = wrapped_layer(new_x)
        # Check the output has the expected shape
        self.assertAllEqual(original_out.get_shape(), new_out.get_shape())
        new_weights, new_biases = wrapped_layer.get_weights()
        # Check the remaining weights are same
        self.assertAllEqual(old_weights[Ellipsis, 1:n_features:2, :], new_weights)
        # Check the remaining biases are same
        self.assertAllEqual(old_biases, new_biases)

    def test_get_config_from_config(self):
        x = tf.random.uniform((2, 10, 2))
        layer = tf.keras.layers.Dense(10, activation="relu")
        to_init_out = layer(x)
        wrapped_layer = GP_ModelParser.parse_layer(layer)
        original_output = wrapped_layer(x)
        weights = wrapped_layer.get_weights()
        layer_config = wrapped_layer.get_config()
        layer_config.update(dict(weights=weights))
        wrapped_layer_copy = wrapped_layer.__class__.from_config(layer_config)
        new_output = wrapped_layer_copy(x)
        self.assertAllEqual(original_output, new_output)
        self.assertIsNot(layer, wrapped_layer_copy)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True)


class LayerTupleGrowPrunTest(parameterized.TestCase, tf.test.TestCase):
    """ Tests growing and pruning with layer tuple."""
    def setUp(self):
        super().setUp()
        global optimizer
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True)

    @parameterized.named_parameters(
        # Last layer is output layer and can't be altered
        ('first_last', (tf.keras.layers.Conv2D(10, 3), tf.keras.layers.Dense(10),
                        tf.keras.layers.Dense(10)), (3, 5, 5, 4)),
        ('first_last_intermediate', (tf.keras.layers.Conv2D(10, 3), utils.ScalingReg(),
                                     tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(10, 3),
                                     tf.keras.layers.Dense(10)),
         (3, 5, 5, 4)),
    )
    def test_weights_grow(self, layer_tuple, input_shape):
        x_input = tf.keras.Input(input_shape[1:])
        x_temp = x_input
        for layer in layer_tuple:
            x_temp = layer(x_temp)

        input_model = tf.keras.Model(inputs=x_input, outputs=x_temp)
        gp_model = structure_adaption.parse_model(input_model)
        orig_units_count = gp_model.internal_model.layers[1].get_units_count()

        def compile_fn():
            gp_model(tf.keras.Input(input_shape[1:]))

        compile_fn()

        seq_branches = gp_model.sequential_branches
        growing_pruning_tuples = gp_model.grow_prun_tuples
        self.assertEqual(1, len(seq_branches))
        self.assertEqual(1, len(growing_pruning_tuples))
        gp_tuple = growing_pruning_tuples[0]

        old_weights = [layer.weights for layer in gp_tuple.layer_tuple]
        old_weights_flat = tf.nest.flatten(old_weights)
        # Important to create own slots because optimizer is not used otherwise
        optimizer._create_slots(old_weights_flat)
        # With SGD only slot name: 'momentum'. Newly created slots should be zero
        slot_names = optimizer.get_slot_names()
        old_slot_vars = []
        for weight in old_weights_flat:
            slot_var = optimizer.get_slot(weight, 'momentum')
            self.assertEqual(0, tf.math.count_nonzero(slot_var))
            old_slot_vars.append(slot_var)
        init_dict = dict(first=[tf.keras.initializers.Ones(), tf.keras.initializers.Zeros()],
                         intermediate=[None, None], last=[tf.keras.initializers.Ones(), tf.keras.initializers.Zeros()])
        gp_model.grow(gp_tuple, 3, init_dict, optimizer, compile_fn)
        gp_tuple = gp_model.grow_prun_tuples[0]  # Need to get new tuples. The old ones are invalid after retracing
        new_weights_pre_call = [layer.weights for layer in gp_tuple.layer_tuple]
        new_weights_pre_call_flat = tf.nest.flatten(new_weights_pre_call)
        # Make actual copy not just copying references
        new_weights_pre_call_flat = [tf.identity(var) for var in new_weights_pre_call_flat]
        x = tf.random.uniform(input_shape)
        output = gp_model(x)
        new_weights_post_call = [layer.weights for layer in gp_tuple.layer_tuple]
        new_weights_post_call_flat = tf.nest.flatten(new_weights_post_call)
        # check if the old weights are preserved
        # first layer
        # kernel
        self.assertAllEqual(new_weights_pre_call[0][0][Ellipsis, :orig_units_count], old_weights[0][0])
        # bias
        self.assertAllEqual(new_weights_pre_call[0][1][:orig_units_count], old_weights[0][1])
        # intermediate layers
        old_weights_intermediate_flat = tf.nest.flatten(old_weights[1:-1])
        new_weights_intermediate_flat = tf.nest.flatten(new_weights_pre_call[1:-1])
        for old_weight, new_weight in zip(old_weights_intermediate_flat, new_weights_intermediate_flat):
            self.assertAllEqual(old_weight, new_weight[:orig_units_count])
        # last layer
        # kernel
        self.assertAllEqual(new_weights_pre_call[-1][0][Ellipsis, :orig_units_count, :], old_weights[-1][0])
        # bias
        self.assertAllEqual(new_weights_pre_call[-1][1], old_weights[-1][1])
        # check that the call doesn't change the weights
        self.assertEqual(len(new_weights_pre_call_flat), len(new_weights_post_call_flat))
        for i in range(len(new_weights_pre_call_flat)):
            self.assertAllEqual(new_weights_pre_call_flat[i], new_weights_post_call_flat[i])

    @parameterized.named_parameters(
        ('first_last', (tf.keras.layers.Conv2D(10, 3), tf.keras.layers.Dense(10), tf.keras.layers.Dense(10)), (3, 5, 5, 4)),
        ('first_last_intermediate', (tf.keras.layers.Conv2D(10, 3), utils.ScalingReg(),
                                     tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(10, 3),
                                     tf.keras.layers.Dense(10)),
         (3, 5, 5, 4)),
    )
    def test_weights_prun(self, layer_tuple, input_shape):
        x_input = tf.keras.Input(input_shape[1:])
        x_temp = x_input
        for layer in layer_tuple:
            x_temp = layer(x_temp)

        input_model = tf.keras.Model(inputs=x_input, outputs=x_temp)
        gp_model = structure_adaption.parse_model(input_model)
        orig_units_count = gp_model.internal_model.layers[1].get_units_count()
        print('orig units count: ' + str(orig_units_count))
        def compile_fn():
            gp_model(tf.keras.Input(input_shape[1:]))

        compile_fn()

        seq_branches = gp_model.sequential_branches
        growing_pruning_tuples = gp_model.grow_prun_tuples
        self.assertEqual(1, len(seq_branches))
        self.assertEqual(1, len(growing_pruning_tuples))
        gp_tuple = growing_pruning_tuples[0]
        old_weights = [layer.weights for layer in gp_tuple.layer_tuple]
        old_weights_flat = tf.nest.flatten(old_weights)
        # Important to create own slots because optimizer is not used otherwise
        optimizer._create_slots(old_weights_flat)
        # With SGD only slot name: 'momentum'. Newly created slots should be zero
        slot_names = optimizer.get_slot_names()
        old_slot_vars = []
        for weight in old_weights_flat:
            slot_var = optimizer.get_slot(weight, 'momentum')
            self.assertEqual(0, tf.math.count_nonzero(slot_var))
            old_slot_vars.append(slot_var)

        gp_model.prun(gp_tuple, [0, 2, 4, 6, 8], optimizer, compile_fn)
        gp_tuple = gp_model.grow_prun_tuples[0]  # Need to get new tuples. The old ones are invalid after retracing
        new_weights_pre_call = [layer.weights for layer in gp_tuple.layer_tuple]
        new_weights_pre_call_flat = tf.nest.flatten(new_weights_pre_call)
        # Make actual copy not just copying references
        new_weights_pre_call_flat = [tf.identity(var) for var in new_weights_pre_call_flat]
        x = tf.random.uniform(input_shape)
        output = gp_model(x)
        new_weights_post_call = [layer.get_weights() for layer in gp_tuple.layer_tuple]
        new_weights_post_call_flat = tf.nest.flatten(new_weights_post_call)
        # check if the old weights are preserved
        # first layer
        # kernel
        self.assertAllEqual(new_weights_pre_call[0][0], old_weights[0][0][Ellipsis, 1:orig_units_count:2])
        # bias
        self.assertAllEqual(new_weights_pre_call[0][1], old_weights[0][1][1:orig_units_count:2])
        # intermediate layers
        old_weights_intermediate_flat = tf.nest.flatten(old_weights[1:-1])
        new_weights_intermediate_flat = tf.nest.flatten(new_weights_pre_call[1:-1])
        for old_weight, new_weight in zip(old_weights_intermediate_flat, new_weights_intermediate_flat):
            self.assertAllEqual(old_weight[1:orig_units_count:2], new_weight)
        # last layer
        # kernel
        self.assertAllEqual(new_weights_pre_call[-1][0], old_weights[-1][0][Ellipsis, 1:orig_units_count:2, :])
        # bias
        self.assertAllEqual(new_weights_pre_call[-1][1], old_weights[-1][1])
        # check that the call doesn't change the weights
        self.assertEqual(len(new_weights_pre_call_flat), len(new_weights_post_call_flat))
        for i in range(len(new_weights_pre_call_flat)):
            self.assertAllEqual(new_weights_pre_call_flat[i], new_weights_post_call_flat[i])

    @parameterized.named_parameters(
        ('first_last', (tf.keras.layers.Conv2D(10, 3), tf.keras.layers.Dense(10), tf.keras.layers.Dense(10)),
         (3, 5, 5, 4)),
    )
    def test_slot_vars_grow(self, layer_tuple, input_shape):
        x_input = tf.keras.Input(input_shape[1:])
        x_temp = x_input
        for layer in layer_tuple:
            x_temp = layer(x_temp)

        input_model = tf.keras.Model(inputs=x_input, outputs=x_temp)
        gp_model = structure_adaption.parse_model(input_model)
        orig_units_count = gp_model.internal_model.layers[1].get_units_count()

        def compile_fn():
            gp_model(tf.keras.Input(input_shape[1:]))

        compile_fn()

        seq_branches = gp_model.sequential_branches
        growing_pruning_tuples = gp_model.grow_prun_tuples
        self.assertEqual(1, len(seq_branches))
        self.assertEqual(1, len(growing_pruning_tuples))
        gp_tuple = growing_pruning_tuples[0]

        old_weights = [layer.weights for layer in gp_tuple.layer_tuple]
        old_weights_flat = tf.nest.flatten(old_weights)
        # Do one minimize step
        loss = lambda: sum([tf.math.reduce_sum(weight) for weight in old_weights_flat])
        optimizer.minimize(loss, old_weights_flat)
        old_slot_vars = []
        for weight in old_weights_flat:
            slot_var = optimizer.get_slot(weight, 'momentum')
            old_slot_vars.append(slot_var)

        # Get old slot vars
        old_kernel_slot_first = tf.identity(optimizer.get_slot(old_weights[0][0], 'momentum'))
        old_bias_slot_first = tf.identity(optimizer.get_slot(old_weights[0][1], 'momentum'))
        old_kernel_slot_last = tf.identity(optimizer.get_slot(old_weights[1][0], 'momentum'))
        old_bias_slot_last = tf.identity(optimizer.get_slot(old_weights[1][1], 'momentum'))

        init_dict = dict(first=[tf.keras.initializers.Ones(), tf.keras.initializers.Zeros()],
                         intermediate=[None, None],
                         last=[tf.keras.initializers.Ones(), tf.keras.initializers.Zeros()])
        gp_model.grow(gp_tuple, 3, init_dict, optimizer, compile_fn)
        gp_tuple = gp_model.grow_prun_tuples[0]
        new_weights_pre_call = [layer.weights for layer in gp_tuple.layer_tuple]
        new_weights_pre_call_flat = tf.nest.flatten(new_weights_pre_call)
        x = tf.random.uniform(input_shape)
        output = gp_model(x)
        new_weights_post_call = [layer.weights for layer in gp_tuple.layer_tuple]
        new_weights_post_call_flat = tf.nest.flatten(new_weights_post_call)
        # check if the old slot variables are preserved
        # first layer
        # kernel
        new_kernel_slot_first = optimizer.get_slot(new_weights_pre_call[0][0], 'momentum')
        self.assertAllEqual(new_kernel_slot_first[Ellipsis, :orig_units_count], old_kernel_slot_first)
        # bias
        new_bias_slot_first = optimizer.get_slot(new_weights_pre_call[0][1], 'momentum')
        self.assertAllEqual(new_bias_slot_first[:orig_units_count], old_bias_slot_first)
        # last layer
        # kernel
        new_kernel_slot_last = optimizer.get_slot(new_weights_pre_call[1][0], 'momentum')
        self.assertAllEqual(new_kernel_slot_last[Ellipsis, :orig_units_count, :], old_kernel_slot_last)
        # bias
        new_bias_slot_last = optimizer.get_slot(new_weights_pre_call[1][1], 'momentum')
        self.assertAllEqual(new_bias_slot_last, old_bias_slot_last)
        # check that the call doesn't change the weights
        self.assertEqual(len(new_weights_pre_call_flat), len(new_weights_post_call_flat))
        for i in range(len(new_weights_pre_call_flat)):
            slot_var_pre = optimizer.get_slot(new_weights_pre_call_flat[i], 'momentum')
            slot_var_post = optimizer.get_slot(new_weights_post_call_flat[i], 'momentum')
            self.assertAllEqual(slot_var_pre, slot_var_post)

    @parameterized.named_parameters(
        ('first_last', (tf.keras.layers.Conv2D(10, 3), tf.keras.layers.Dense(10), tf.keras.layers.Dense(10)),
         (3, 5, 5, 4)),
    )
    def test_slot_vars_prun(self, layer_tuple, input_shape):
        x_input = tf.keras.Input(input_shape[1:])
        x_temp = x_input
        for layer in layer_tuple:
            x_temp = layer(x_temp)

        input_model = tf.keras.Model(inputs=x_input, outputs=x_temp)
        gp_model = structure_adaption.parse_model(input_model)
        orig_units_count = gp_model.internal_model.layers[1].get_units_count()

        def compile_fn():
            gp_model(tf.keras.Input(input_shape[1:]))

        compile_fn()

        seq_branches = gp_model.sequential_branches
        growing_pruning_tuples = gp_model.grow_prun_tuples
        self.assertEqual(1, len(seq_branches))
        self.assertEqual(1, len(growing_pruning_tuples))
        gp_tuple = growing_pruning_tuples[0]
        old_weights = [layer.weights for layer in gp_tuple.layer_tuple]
        old_weights_flat = tf.nest.flatten(old_weights)
        # Do one minimize step
        loss = lambda: sum([tf.math.reduce_sum(weight) for weight in old_weights_flat])
        optimizer.minimize(loss, old_weights_flat)
        old_slot_vars = []
        for weight in old_weights_flat:
            slot_var = optimizer.get_slot(weight, 'momentum')
            old_slot_vars.append(slot_var)

        # Get old slot vars
        old_kernel_slot_first = tf.identity(optimizer.get_slot(old_weights[0][0], 'momentum'))
        old_bias_slot_first = tf.identity(optimizer.get_slot(old_weights[0][1], 'momentum'))
        old_kernel_slot_last = tf.identity(optimizer.get_slot(old_weights[1][0], 'momentum'))
        old_bias_slot_last = tf.identity(optimizer.get_slot(old_weights[1][1], 'momentum'))

        gp_model.prun(gp_tuple, [0, 2, 4, 6, 8], optimizer, compile_fn)
        gp_tuple = gp_model.grow_prun_tuples[0]
        new_weights_pre_call = [layer.weights for layer in gp_tuple.layer_tuple]
        new_weights_pre_call_flat = tf.nest.flatten(new_weights_pre_call)
        x = tf.random.uniform(input_shape)
        output = gp_model(x)
        new_weights_post_call = [layer.weights for layer in gp_tuple.layer_tuple]
        new_weights_post_call_flat = tf.nest.flatten(new_weights_post_call)
        # check if the old slot variables are preserved
        # first layer
        # kernel
        new_kernel_slot_first = optimizer.get_slot(new_weights_pre_call[0][0], 'momentum')
        self.assertAllEqual(new_kernel_slot_first, old_kernel_slot_first[Ellipsis, 1:orig_units_count:2])
        # bias
        new_bias_slot_first = optimizer.get_slot(new_weights_pre_call[0][1], 'momentum')
        self.assertAllEqual(new_bias_slot_first, old_bias_slot_first[1:orig_units_count:2])
        # last layer
        # kernel
        new_kernel_slot_last = optimizer.get_slot(new_weights_pre_call[1][0], 'momentum')
        self.assertAllEqual(new_kernel_slot_last, old_kernel_slot_last[Ellipsis, 1:orig_units_count:2, :])
        # bias
        new_bias_slot_last = optimizer.get_slot(new_weights_pre_call[1][1], 'momentum')
        self.assertAllEqual(new_bias_slot_last, old_bias_slot_last)
        # check that the call doesn't change the weights
        self.assertEqual(len(new_weights_pre_call_flat), len(new_weights_post_call_flat))
        for i in range(len(new_weights_pre_call_flat)):
            slot_var_pre = optimizer.get_slot(new_weights_pre_call_flat[i], 'momentum')
            slot_var_post = optimizer.get_slot(new_weights_post_call_flat[i], 'momentum')
            self.assertAllEqual(slot_var_pre, slot_var_post)


def print_branches(branches):
    for i, branch in enumerate(branches):
        print(f'Branch no {i}:')
        for layer in branch.layers:
            print(layer.name + ', ', end='')
        print('')


def print_parallel_branches(parallel_branches):
    for i, parallel_branch in enumerate(parallel_branches):
        print(f'Parallel Branch no {i}:')
        for contained_branch in parallel_branch.branches:
            for layer in contained_branch.layers:
                print(layer.name + ', ', end='')
            print(' ;  ', end='')
        print('')


class GrowingPruningModelParserTest(tf.test.TestCase):

    def test_parse_layer(self):
        x = tf.random.uniform((2, 10, 2))
        layer = tf.keras.layers.Dense(10, activation='relu')
        original_output = layer(x)
        parser_non_gp = structure_adaption.GrowingPruningModelParser()
        parsed_layer_non_gp = parser_non_gp.parse_layer(layer)
        non_gp_output = parsed_layer_non_gp(x)
        parser_gp_gradmax = structure_adaption.GrowingPruningModelParser(supported_layers_config=supp_lay_conf)
        parsed_layer_gp = parser_gp_gradmax.parse_layer(layer)
        gp_output = parsed_layer_gp(x)
        self.assertAllClose(original_output, non_gp_output)
        self.assertAllClose(original_output, gp_output)

    def test_get_seq_par_branches0(self):
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x1_l = tf.keras.layers.Dense(10, name='x1'); x1 = x1_l(inputs)
        x21_l = tf.keras.layers.Dense(10, name='x21'); x21 = x21_l(x1)
        x22_l = tf.keras.layers.Dense(10, name='x22'); x22 = x22_l(x1)
        x31_l = tf.keras.layers.Dense(10, name='x31'); x31 = x31_l(x22)
        x32_l = tf.keras.layers.Dense(10, name='x32'); x32 = x32_l(x22)
        x4_l = tf.keras.layers.Add(name='x4'); x4 = x4_l([x21, x31, x32])
        x51_l = tf.keras.layers.Dense(10, name='x51'); x51 = x51_l(x4)
        x52_l = tf.keras.layers.Dense(10, name='x52'); x52 = x52_l(x4)
        x6_l = tf.keras.layers.Add(name='x6'); outputs = x6_l([x51, x52])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x0_l = model._input_layers[0]
        branches = structure_adaption.GrowingPruningModelParser._get_seq_branches(model)
        parallel_branches = structure_adaption.GrowingPruningModelParser._get_par_branches(model, branches)
        branch0 = structure_adaption.Branch([x0_l, x1_l])
        branch1 = structure_adaption.Branch([x1_l, x21_l, x4_l])
        branch2 = structure_adaption.Branch([x4_l, x51_l, x6_l])
        branch3 = structure_adaption.Branch([x4_l, x52_l, x6_l])
        branch4 = structure_adaption.Branch([x1_l, x22_l])
        branch5 = structure_adaption.Branch([x22_l, x31_l, x4_l])
        branch6 = structure_adaption.Branch([x22_l, x32_l, x4_l])
        expected_branches = [branch0, branch1, branch2, branch3, branch4, branch5, branch6]
        par_branch0 = structure_adaption.ParallelBranch([x4_l, x51_l, x52_l, x6_l], [])
        par_branch1 = structure_adaption.ParallelBranch([x22_l, x31_l, x32_l, x4_l], [])
        par_branch2 = structure_adaption.ParallelBranch([x1_l, x21_l, x22_l, x31_l, x32_l, x4_l], [])
        expected_par_branches = [par_branch0, par_branch1, par_branch2]
        self.assertSetEqual(set(expected_branches), set(branches))
        self.assertSetEqual(set(expected_par_branches), set(parallel_branches))

    def test_get_seq_par_branches1(self):
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x1_l = tf.keras.layers.Dense(10, name='x1'); x1 = x1_l(inputs)
        x21_l = tf.keras.layers.Dense(10, name='x21'); x21 = x21_l(x1)
        x22_l = tf.keras.layers.Dense(10, name='x22'); x22 = x22_l(x1)
        x31_l = tf.keras.layers.Dense(10, name='x31'); x31 = x31_l(x22)
        x32_l = tf.keras.layers.Dense(10, name='x32'); x32 = x32_l(x22)
        x4_l = tf.keras.layers.Add(name='x4'); x4 = x4_l([x31, x32])
        x5_l = tf.keras.layers.Add(name='x5'); x5 = x5_l([x21, x4])
        x61_l = tf.keras.layers.Dense(10, name='x61'); x61 = x61_l(x5)
        x62_l = tf.keras.layers.Dense(10, name='x62'); x62 = x62_l(x5)
        x7_l = tf.keras.layers.Add(name='x7'); outputs = x7_l([x61, x62])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x0_l = model._input_layers[0]
        branches = structure_adaption.GrowingPruningModelParser._get_seq_branches(model)
        parallel_branches = structure_adaption.GrowingPruningModelParser._get_par_branches(model, branches)
        branch0 = structure_adaption.Branch([x0_l, x1_l])
        branch1 = structure_adaption.Branch([x1_l, x21_l, x5_l])
        branch2 = structure_adaption.Branch([x5_l, x61_l, x7_l])
        branch3 = structure_adaption.Branch([x5_l, x62_l, x7_l])
        branch4 = structure_adaption.Branch([x1_l, x22_l])
        branch5 = structure_adaption.Branch([x22_l, x31_l, x4_l])
        branch6 = structure_adaption.Branch([x22_l, x32_l, x4_l])
        branch7 = structure_adaption.Branch([x4_l, x5_l])
        expected_branches = [branch0, branch1, branch2, branch3, branch4, branch5, branch6, branch7]
        par_branch0 = structure_adaption.ParallelBranch([x5_l, x61_l, x62_l, x7_l], [])
        par_branch1 = structure_adaption.ParallelBranch([x22_l, x31_l, x32_l, x4_l], [])
        par_branch2 = structure_adaption.ParallelBranch([x1_l, x21_l, x22_l, x31_l, x32_l, x4_l, x5_l], [])
        expected_par_branches = [par_branch0, par_branch1, par_branch2]
        self.assertSetEqual(set(expected_branches), set(branches))
        self.assertSetEqual(set(expected_par_branches), set(parallel_branches))

    def test_get_seq_par_branches2(self):
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x1_l = tf.keras.layers.Dense(10, name='x1'); x1 = x1_l(inputs)
        x21_l = tf.keras.layers.Dense(10, name='x21'); x21 = x21_l(x1)
        x22_l = tf.keras.layers.Dense(10, name='x22'); x22 = x22_l(x1)
        x23_l = tf.keras.layers.Dense(10, name='x23'); x23 = x23_l(x1)
        x31_l = tf.keras.layers.Dense(10, name='x31'); x31 = x31_l(x22)
        x32_l = tf.keras.layers.Dense(10, name='x32'); x32 = x32_l(x23)
        x4_l = tf.keras.layers.Add(name='x4'); x4 = x4_l([x31, x32])
        x5_l = tf.keras.layers.Add(name='x5'); x5 = x5_l([x21, x4])
        x61_l = tf.keras.layers.Dense(10, name='x61'); x61 = x61_l(x5)
        x62_l = tf.keras.layers.Dense(10, name='x62'); x62 = x62_l(x5)
        x7_l = tf.keras.layers.Add(name='x7'); outputs = x7_l([x61, x62])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x0_l = model._input_layers[0]
        branches = structure_adaption.GrowingPruningModelParser._get_seq_branches(model)
        parallel_branches = structure_adaption.GrowingPruningModelParser._get_par_branches(model, branches)
        branch0 = structure_adaption.Branch([x0_l, x1_l])
        branch1 = structure_adaption.Branch([x1_l, x21_l, x5_l])
        branch2 = structure_adaption.Branch([x5_l, x61_l, x7_l])
        branch3 = structure_adaption.Branch([x5_l, x62_l, x7_l])
        branch4 = structure_adaption.Branch([x1_l, x22_l, x31_l, x4_l])
        branch5 = structure_adaption.Branch([x1_l, x23_l, x32_l, x4_l])
        branch6 = structure_adaption.Branch([x4_l, x5_l])
        expected_branches = [branch0, branch1, branch2, branch3, branch4, branch5, branch6]
        par_branch0 = structure_adaption.ParallelBranch([x5_l, x61_l, x62_l, x7_l], [])
        par_branch1 = structure_adaption.ParallelBranch([x1_l, x22_l, x23_l, x31_l, x32_l, x4_l], [])
        par_branch2 = structure_adaption.ParallelBranch([x1_l, x21_l, x22_l, x23_l, x31_l, x32_l, x4_l, x5_l], [])
        expected_par_branches = [par_branch0, par_branch1, par_branch2]
        self.assertSetEqual(set(expected_branches), set(branches))
        self.assertSetEqual(set(expected_par_branches), set(parallel_branches))

    def test_get_seq_par_branches3(self):
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x1_l = tf.keras.layers.Dense(10, name='x1'); x1 = x1_l(inputs)
        x21_l = tf.keras.layers.Dense(10, name='x21'); x21 = x21_l(x1)
        x22_l = tf.keras.layers.Dense(10, name='x22'); x22 = x22_l(x1)
        x31_l = tf.keras.layers.Dense(10, name='x31'); x31 = x31_l(x22)
        x32_l = tf.keras.layers.Dense(10, name='x32'); x32 = x32_l(x22)
        x41_l = tf.keras.layers.Dense(10, name='x41'); x41 = x41_l(x31)
        x42_l = tf.keras.layers.Dense(10, name='x42'); x42 = x42_l(x32)
        x5_l = tf.keras.layers.Add(name='x5'); x5 = x5_l([x21, x41, x42])
        x6_l = tf.keras.layers.Dense(10, name='x6'); outputs = x6_l(x5)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x0_l = model._input_layers[0]
        branches = structure_adaption.GrowingPruningModelParser._get_seq_branches(model)
        parallel_branches = structure_adaption.GrowingPruningModelParser._get_par_branches(model, branches)
        branch0 = structure_adaption.Branch([x0_l, x1_l])
        branch1 = structure_adaption.Branch([x1_l, x21_l, x5_l])
        branch2 = structure_adaption.Branch([x5_l, x6_l])
        branch3 = structure_adaption.Branch([x1_l, x22_l])
        branch4 = structure_adaption.Branch([x22_l, x31_l, x41_l, x5_l])
        branch5 = structure_adaption.Branch([x22_l, x32_l, x42_l, x5_l])
        expected_branches = [branch0, branch1, branch2, branch3, branch4, branch5]
        par_branch0 = structure_adaption.ParallelBranch([x22_l, x31_l, x32_l, x41_l, x42_l, x5_l], [])
        par_branch1 = structure_adaption.ParallelBranch([x1_l, x21_l, x22_l, x31_l, x32_l, x41_l, x42_l, x5_l], [])
        expected_par_branches = [par_branch0, par_branch1]
        self.assertSetEqual(set(expected_branches), set(branches))
        self.assertSetEqual(set(expected_par_branches), set(parallel_branches))

    def test_get_seq_par_branches4(self):
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x1_l = tf.keras.layers.Dense(10, name='x1'); x1 = x1_l(inputs)
        x2_l = tf.keras.layers.Dense(10, name='x2'); x2 = x2_l(x1)
        x3_l = tf.keras.layers.Dense(10, name='x3'); x3 = x3_l(x2)
        x4_l = tf.keras.layers.Add(name='x4'); x4 = x4_l([x1, x3])
        x5_l = tf.keras.layers.Dense(10, name='x5'); x5 = x5_l(x4)
        x6_l = tf.keras.layers.Add(name='x6'); x6 = x6_l([x4, x5])
        x7_l = tf.keras.layers.Dense(10, name='x7'); outputs = x7_l(x6)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x0_l = model._input_layers[0]
        branches = structure_adaption.GrowingPruningModelParser._get_seq_branches(model)
        parallel_branches = structure_adaption.GrowingPruningModelParser._get_par_branches(model, branches)
        branch0 = structure_adaption.Branch([x0_l, x1_l])
        branch1 = structure_adaption.Branch([x1_l, x4_l])
        branch2 = structure_adaption.Branch([x4_l, x6_l])
        branch3 = structure_adaption.Branch([x6_l, x7_l])
        branch4 = structure_adaption.Branch([x4_l, x5_l, x6_l])
        branch5 = structure_adaption.Branch([x1_l, x2_l, x3_l, x4_l])
        expected_branches = [branch0, branch1, branch2, branch3, branch4, branch5]
        par_branch0 = structure_adaption.ParallelBranch([x4_l, x5_l, x6_l], [])
        par_branch1 = structure_adaption.ParallelBranch([x1_l, x2_l, x3_l, x4_l], [])
        expected_par_branches = [par_branch0, par_branch1]
        self.assertSetEqual(set(expected_branches), set(branches))
        self.assertSetEqual(set(expected_par_branches), set(parallel_branches))

    def test_get_seq_par_branches5(self):
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x11_l = tf.keras.layers.Dense(10, name='x11'); x11 = x11_l(inputs)
        x12_l = tf.keras.layers.Dense(10, name='x12'); x12 = x12_l(inputs)
        x2_l = tf.keras.layers.Add(name='x2'); x2 = x2_l([x11, x12])
        x31_l = tf.keras.layers.Dense(10, name='x31'); x31 = x31_l(x2)
        x32_l = tf.keras.layers.Dense(10, name='x32'); x32 = x32_l(x2)
        x4_l = tf.keras.layers.Add(name='x4'); outputs = x4_l([x31, x32])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x0_l = model._input_layers[0]
        branches = structure_adaption.GrowingPruningModelParser._get_seq_branches(model)
        parallel_branches = structure_adaption.GrowingPruningModelParser._get_par_branches(model, branches)
        branch0 = structure_adaption.Branch([x0_l, x11_l, x2_l])
        branch1 = structure_adaption.Branch([x2_l, x31_l, x4_l])
        branch2 = structure_adaption.Branch([x2_l, x32_l, x4_l])
        branch3 = structure_adaption.Branch([x0_l, x12_l, x2_l])
        expected_branches = [branch0, branch1, branch2, branch3]
        par_branch0 = structure_adaption.ParallelBranch([x0_l, x11_l, x12_l, x2_l], [])
        par_branch1 = structure_adaption.ParallelBranch([x2_l, x31_l, x32_l, x4_l], [])
        expected_par_branches = [par_branch0, par_branch1]
        self.assertSetEqual(set(expected_branches), set(branches))
        self.assertSetEqual(set(expected_par_branches), set(parallel_branches))

    def test_get_seq_par_branches6(self):
        # pure sequential case
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x1_l = tf.keras.layers.Dense(10, name='x1'); x1 = x1_l(inputs)
        x2_l = tf.keras.layers.Dense(10, name='x2'); x2 = x2_l(x1)
        x3_l = tf.keras.layers.Dense(10, name='x3'); x3 = x3_l(x2)
        x4_l = tf.keras.layers.Dense(10, name='x4'); outputs = x4_l(x3)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x0_l = model._input_layers[0]
        branches = structure_adaption.GrowingPruningModelParser._get_seq_branches(model)
        parallel_branches = structure_adaption.GrowingPruningModelParser._get_par_branches(model, branches)
        branch0 = structure_adaption.Branch([x0_l, x1_l, x2_l, x3_l, x4_l])
        expected_branches = [branch0]
        expected_par_branches = []
        self.assertSetEqual(set(expected_branches), set(branches))
        self.assertSetEqual(set(expected_par_branches), set(parallel_branches))

    def test_get_seq_par_branches7(self):
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x1_l = tf.keras.layers.Dense(3, name='x1'); x1 = x1_l(inputs)
        x2_l = tf.keras.layers.Add(name='x2'); outputs = x2_l([x1, inputs])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x0_l = model._input_layers[0]
        branches = structure_adaption.GrowingPruningModelParser._get_seq_branches(model)
        parallel_branches = structure_adaption.GrowingPruningModelParser._get_par_branches(model, branches)
        branch0 = structure_adaption.Branch([x0_l, x2_l])
        branch1 = structure_adaption.Branch([x0_l, x1_l, x2_l])
        expected_branches = [branch0, branch1]
        par_branch0 = structure_adaption.ParallelBranch([x0_l, x1_l, x2_l], [])
        expected_par_branches = [par_branch0]
        self.assertSetEqual(set(expected_branches), set(branches))
        self.assertSetEqual(set(expected_par_branches), set(parallel_branches))


class InternalModelTest(tf.test.TestCase):
    # TODO: Concatenate merge layers not considered!
    def setUp(self):
        super().setUp()
        global optimizer
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True)

    @staticmethod
    def get_layer_names(layers):
        # since the layers will be copied when retracing, identifying by name is the easiest way to compare
        return [layer.name for layer in layers]

    def assertEqualSeqBranchesNames(self, exp_seq_branches_names, seq_branches):
        seq_branches_len = len(seq_branches)
        self.assertEqual(seq_branches_len, len(exp_seq_branches_names))
        seq_branches_count = 0
        seq_branches_names = []

        for seq_branch in seq_branches:
            names_set = set(self.get_layer_names(seq_branch.layers))
            seq_branches_names.append(names_set)

        for i in range(len(exp_seq_branches_names)):
            exp_seq_branches_names[i] = set(exp_seq_branches_names[i])

        for i in range(len(seq_branches_names)):
            for j in range(len(exp_seq_branches_names)):
                if seq_branches_names[i] and exp_seq_branches_names[j] and\
                        not seq_branches_names[i].difference(exp_seq_branches_names[j]):
                    seq_branches_count += 1
                    seq_branches_names[i] = None
                    exp_seq_branches_names[j] = None
                    break

        if seq_branches_len == seq_branches_count:
            return
        else:
            seq_branches_names = [seq_branch_names for seq_branch_names in seq_branches_names if seq_branch_names]
            exp_seq_branches_names = [exp_seq_branch_names for exp_seq_branch_names in exp_seq_branches_names
                                      if exp_seq_branch_names]
            lines = []
            lines.append('Sequential branches don\'t match')
            lines.append('Expected not matching branches: ')
            for exp_seq_branch_names in exp_seq_branches_names:
                lines.append(repr(exp_seq_branch_names))
            lines.append('Actual not matching branches: ')
            for seq_branch_names in seq_branches_names:
                lines.append(repr(seq_branch_names))

            standardMsg = '\n'.join(lines)
            self.fail(standardMsg)

    @staticmethod
    def get_optimizer_slot_dict(optimizer):
        optimizer_slots = optimizer._slots.copy()
        optimizer_slots = [optimizer_slot['momentum'] for optimizer_slot in optimizer_slots.values()]
        optimizer_slots_dict = {optimizer_slot.name: optimizer_slot for optimizer_slot in optimizer_slots}
        return optimizer_slots_dict

    def assertEqualSlotsCarried(self, dict_old, dict_new, ignore_names=None):
        a_length = len(dict_old)
        b_length = len(dict_new)
        if a_length > b_length:
            superset_dict = dict_old
            subset_dict = dict_new
        else:
            superset_dict = dict_new
            subset_dict = dict_old
        keys = list(subset_dict.keys())
        if ignore_names is None:
            ignore_names = []
        else:
            for key in keys:
                for ignore_name in ignore_names:
                    if ignore_name in key:
                        self.assertAllEqual(dict_new[key], tf.zeros_like(dict_new[key]))

        for key in keys:
            for ignore_name in ignore_names:
                if ignore_name in key:
                    break
            else:
                self.assertAllEqual(superset_dict[key], subset_dict[key])

    def test_layer_pair_hash(self):
        edges = set()
        l1 = tf.keras.layers.Dense(units=16, activation='relu', name='l1')
        l2 = tf.keras.layers.Softmax(name='l2')
        layer_pair1 = structure_adaption.GrowingPruningModel.LayerPair(l1, l2)
        edges.add(layer_pair1)
        layer_pair2 = structure_adaption.GrowingPruningModel.LayerPair(l1, l2)
        edges_hash = set()
        edges_hash.add(layer_pair1.__hash__())
        self.assertTrue(layer_pair2.__hash__() in edges_hash)

    def test_layer_recall(self):
        input_model = test_networks.sequential_long()
        layers = input_model.layers
        l2 = layers[2]
        l6 = tf.keras.layers.Dense(units=16, activation='relu', name='x1')
        x6 = l6(l2.output)
        l2 = input_model.layers[2]
        a=5

    def test_remove_0_leave_residual(self):
        input_model = test_networks.sequential_long()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[0]
        prun_branch = structure_adaption.Branch(seq_branch.layers[1:5])
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers, prun_branch_es=prun_branch,
                                                                                                 leave_residual=True)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x1', 'x4', 'x5']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_1_leave_residual(self):
        input_model = test_networks.sequential_long()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[0]
        prun_branch = structure_adaption.Branch(seq_branch.layers[0:6])
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers, prun_branch_es=prun_branch,
                                                                                                 leave_residual=True)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x5']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_2_leave_residual(self):
        input_model = test_networks.non_sequential_simple()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[1]
        prun_branch = seq_branch # gplayers.Branch(seq_branch.layers[0:4])
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers, prun_branch_es=prun_branch,
                                                                                                 leave_residual=True)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(pruned_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x4', 'x5'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x1'], ['x1', 'x4'], ['x1', 'x4'], ['x4', 'x5']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_3_leave_residual(self):
        input_model = test_networks.non_sequential_simple_cut_start()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[0]
        prun_branch = seq_branch
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers, prun_branch_es=prun_branch,
                                                                                                 leave_residual=True)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(pruned_model.layers))
        expected_new_layer_names = {'x0', 'x3', 'x4'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x3'], ['x0', 'x3'], ['x3', 'x4']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_4(self):
        input_model = test_networks.non_sequential_simple()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[1]
        prun_branch = seq_branch
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers, prun_branch_es=prun_branch)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x1', 'x5']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_5(self):
        input_model = test_networks.non_sequential_simple_cut_start()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[2]
        prun_branch = seq_branch
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers, prun_branch_es=prun_branch,
                                                                                                 leave_residual=False)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(pruned_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x2', 'x4'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x1', 'x2', 'x4']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_6_simultaneous(self):
        input_model = test_networks.non_sequential_medium()
        model = structure_adaption.parse_model(input_model)
        seq_branches = model.sequential_branches
        prun_branch1 = seq_branches[6]
        prun_branch2 = seq_branches[4]
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 prun_branch_es=[prun_branch1, prun_branch2], leave_residual=False)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(pruned_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x20', 'x30', 'x4', 'x50', 'x60', 'x8'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x1'], ['x1', 'x20', 'x30', 'x4'], ['x1', 'x4'],
                                       ['x4', 'x50', 'x60', 'x8']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_7_simultaneous(self):
        input_model = test_networks.non_sequential_medium()
        model = structure_adaption.parse_model(input_model)
        seq_branches = model.sequential_branches
        prun_branch1 = seq_branches[6]
        prun_branch2 = seq_branches[5]
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 prun_branch_es=[prun_branch1, prun_branch2], leave_residual=False)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(pruned_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x20', 'x30', 'x50', 'x60', 'x51', 'x61', 'x7', 'x8'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x1', 'x20', 'x30'], ['x30', 'x51', 'x61', 'x7'],
                                       ['x30', 'x50', 'x60', 'x7'], ['x7', 'x8']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_8_nested(self):
        input_model = test_networks.non_sequential_nested()
        model = structure_adaption.parse_model(input_model)
        model_layers = model.internal_model.layers
        indices = [0, 1, 3, 4, 6, 7]
        prun_layers = [model_layers[x] for x in indices]
        prun_branch = structure_adaption.Branch(prun_layers)
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 prun_branch_es=[prun_branch], leave_residual=False)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(pruned_model.layers))
        expected_new_layer_names = {'x0', 'x10', 'x30', 'x5'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x10', 'x30', 'x5']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_9_nested(self):
        input_model = test_networks.non_sequential_nested_cut_branch_start()
        model = structure_adaption.parse_model(input_model)
        model_layers = model.internal_model.layers
        indices = [0, 2, 5]
        prun_layers = [model_layers[x] for x in indices]
        prun_branch = structure_adaption.Branch(prun_layers)
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 prun_branch_es=[prun_branch], leave_residual=False)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(pruned_model.layers))
        expected_new_layer_names = {'x0', 'x10', 'x20', 'x12', 'x3', 'x4'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x10', 'x20', 'x3'], ['x0', 'x12', 'x3'], ['x3', 'x4']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_10_nested_simultaneous(self):
        return
        # TODO doesn't work yet
        input_model = test_networks.non_sequential_complex()
        model = structure_adaption.parse_model(input_model)
        seq_branches = model.sequential_branches
        prun_branch1 = seq_branches[6]
        prun_branch2 = seq_branches[5]
        model_layers = model.internal_model.layers
        indices = [0, 2, 5]
        prun_layers = [model_layers[x] for x in indices]
        prun_branch = structure_adaption.Branch(prun_layers)
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 prun_branch_es=[prun_branch], leave_residual=False)
        pruned_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(pruned_model.layers))
        expected_new_layer_names = {'x0', 'x10', 'x20', 'x12', 'x3', 'x4'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x10', 'x20', 'x3'], ['x0', 'x12', 'x3'], ['x3', 'x4']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_add_0(self):
        input_model = test_networks.sequential_short()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[0]
        start_layer = seq_branch.layers[1]
        end_layer = seq_branch.layers[2]
        l4 = tf.keras.layers.Dense(units=16, activation='relu', name='x4')
        x4 = l4(start_layer.output)
        insert_branch = structure_adaption.InsertBranch([start_layer, l4], end_layer)
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 grow_branch_es=insert_branch, insert_sequential=True)
        grown_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(grown_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x2', 'x3', 'x4'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(grown_model)
        expected_seq_branches_names = [['x0', 'x1', 'x4', 'x2', 'x3']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_add_1(self):
        input_model = test_networks.sequential_short()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[0]
        start_layer = seq_branch.layers[0]
        end_layer = seq_branch.layers[2]
        l4 = tf.keras.layers.Dense(units=16, activation='relu', name='x4')
        x4 = l4(start_layer.output)
        insert_branch = structure_adaption.InsertBranch([start_layer, l4], end_layer)
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 grow_branch_es=insert_branch, insert_sequential=False)
        grown_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(grown_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x2', 'x3', 'x4', 'new_add_node0'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(grown_model)
        expected_seq_branches_names = [['x0', 'x1', 'new_add_node0'], ['x0', 'x4', 'new_add_node0'], ['new_add_node0', 'x2', 'x3']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_add_2(self):
        input_model = test_networks.sequential_short()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[0]
        start_layer = seq_branch.layers[0]
        end_layer = seq_branch.layers[2]
        l4 = tf.keras.layers.Dense(units=16, activation='relu', name='x4')
        x4 = l4(start_layer.output)
        l5 = tf.keras.layers.Dense(units=16, activation='relu', name='x5')
        x5 = l5(x4)
        insert_branch = structure_adaption.InsertBranch([start_layer, l5], end_layer)
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 grow_branch_es=insert_branch, insert_sequential=False)
        grown_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(grown_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'new_add_node0'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(grown_model)
        expected_seq_branches_names = [['x0', 'x1', 'new_add_node0'], ['x0', 'x4', 'x5', 'new_add_node0'],
                                       ['new_add_node0', 'x2', 'x3']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_add_3(self):
        input_model = test_networks.sequential_short()
        model = structure_adaption.parse_model(input_model)
        seq_branch = model.sequential_branches[0]
        start_layer = seq_branch.layers[0]
        end_layer = seq_branch.layers[3]
        l4 = tf.keras.layers.Dense(units=16, activation='relu', name='x4')
        x4 = l4(start_layer.output)
        l5 = tf.keras.layers.Dense(units=16, activation='relu', name='x5')
        x5 = l5(start_layer.output)
        l6 = tf.keras.layers.Add(name='x6')
        x6 = l6([x4, x5])
        insert_branch = structure_adaption.InsertBranch([start_layer, l4, l5, l6], end_layer)
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 grow_branch_es=insert_branch,
                                                                                                 insert_sequential=False)
        grown_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(grown_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'new_add_node0'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(grown_model)
        expected_seq_branches_names = [['x0', 'x1', 'x2', 'new_add_node0'], ['x0', 'x4', 'x6'], ['x0', 'x5', 'x6'],
                                       ['x6', 'new_add_node0'], ['new_add_node0', 'x3']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_add_4(self):
        input_model = test_networks.non_sequential_simple()
        model = structure_adaption.parse_model(input_model)
        layers = model.internal_model.layers

        start_layer = layers[1]
        end_layer = layers[4]
        l6 = tf.keras.layers.Dense(units=16, activation='relu', name='x6')
        x6 = l6(start_layer.output)
        l7 = tf.keras.layers.Dense(units=16, activation='relu', name='x7')
        x7 = l7(x6)
        insert_branch = structure_adaption.InsertBranch([start_layer, l6, l7], end_layer)
        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 grow_branch_es=insert_branch,
                                                                                                 insert_sequential=False)
        grown_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(grown_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(grown_model)
        expected_seq_branches_names = [['x0', 'x1'], ['x1', 'x2', 'x3', 'x4'], ['x1', 'x4'], ['x1', 'x6', 'x7', 'x4'],
                                       ['x4', 'x5']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_add_5(self):
        return
        # TODO: feature not implemented yet
        input_model = test_networks.non_sequential_simple_cut_start()
        model = structure_adaption.parse_model(input_model)
        layers = model.internal_model.layers

        start_layer = layers[0]
        end_layer = layers[3]

        l5 = tf.keras.layers.Dense(units=16, activation='relu', name='x5')
        x5 = l5(start_layer.output)
        l6 = tf.keras.layers.Dense(units=16, activation='relu', name='x6')
        x6 = l6(x5)
        insert_branch1 = structure_adaption.InsertBranch([start_layer, l5, l6], end_layer)

        l7 = tf.keras.layers.Dense(units=16, activation='relu', name='x7')
        x7 = l7(start_layer.output)
        l8 = tf.keras.layers.Dense(units=16, activation='relu', name='x8')
        x8 = l8(x7)
        insert_branch2 = structure_adaption.InsertBranch([start_layer, l7, l8], end_layer)

        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 grow_branch_es=[insert_branch1,
                                                                                                       insert_branch2],
                                                                                                 insert_sequential=False)
        grown_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(grown_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(grown_model)
        expected_seq_branches_names = [['x0', 'x1', 'x2', 'x3'], ['x0', 'x3'], ['x0', 'x5', 'x6', 'x3'],
                                       ['x0', 'x7', 'x8', 'x3'], ['x3', 'x4']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_add_6(self):
        input_model = test_networks.non_sequential_simple()
        model = structure_adaption.parse_model(input_model)
        layers = model.internal_model.layers

        start_layer = layers[1]
        end_layer = layers[4]
        l6 = tf.keras.layers.Dense(units=16, activation='relu', name='x6')
        x6 = l6(start_layer.output)
        l7 = tf.keras.layers.Dense(units=16, activation='relu', name='x7')
        x7 = l7(x6)
        insert_branch1 = structure_adaption.InsertBranch([start_layer, l6, l7], end_layer)

        start_layer = layers[4]
        end_layer = layers[5]
        l8 = tf.keras.layers.Dense(units=16, activation='relu', name='x8')
        x8 = l8(start_layer.output)
        insert_branch2 = structure_adaption.InsertBranch([start_layer, l8], end_layer)

        inputs = model.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 grow_branch_es=[insert_branch1,
                                                                                                       insert_branch2],
                                                                                                 insert_sequential=False)
        grown_model = tf.keras.Model(inputs=inputs[0].output, outputs=outputs[0].output)
        new_layer_names = set(self.get_layer_names(grown_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'new_add_node0'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(grown_model)
        expected_seq_branches_names = [['x0', 'x1'], ['x1', 'x2', 'x3', 'x4'], ['x1', 'x4'], ['x1', 'x6', 'x7', 'x4'],
                                       ['x4', 'new_add_node0'], ['x4', 'x8', 'new_add_node0'], ['new_add_node0', 'x5']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def test_remove_add_carry_optimizer_0(self):
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x1_l = tf.keras.layers.Dense(6, name='x1'); x1 = x1_l(inputs)
        x2_l = tf.keras.layers.Dense(8, name='x2'); x2 = x2_l(x1)
        x3_l = tf.keras.layers.Dense(6, name='x3'); x3 = x3_l(x2)
        x4_l = tf.keras.layers.Add(name='x4'); x4 = x4_l([x1, x3])
        x5 = tf.keras.layers.Dense(6, name='x5')(x4)
        x6 = tf.keras.layers.Add(name='x6')([x4, x5])
        outputs = tf.keras.layers.Dense(14, name='x7')(x6)
        branch = structure_adaption.Branch([x1_l, x2_l, x3_l, x4_l])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model_weights = model.trainable_variables
        optimizer._create_slots(model_weights)
        # run one optimization step, so slots are not zero:
        loss = lambda: sum([tf.math.reduce_sum(weight) for weight in model_weights])
        optimizer.minimize(loss, model_weights)
        optimizer_slots_old_dict = self.get_optimizer_slot_dict(optimizer)
        #prun
        inputs = model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs, outputs] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                 prun_branch_es=branch,
                                                                                                 skip_mismatch=True)

        pruned_model = tf.keras.Model(name=model.name, inputs=inputs[0].output, outputs=outputs[0].output)

        pruned_model_weights = pruned_model.trainable_variables
        structure_adaption.GrowingPruningModel.maybe_carry_optimizer_slots(optimizer, model_weights,
                                                                           pruned_model_weights, ignore_names=['x5', 'x6'])
        optimizer_slots_prun_dict = self.get_optimizer_slot_dict(optimizer)
        self.assertEqualSlotsCarried(optimizer_slots_old_dict, optimizer_slots_prun_dict, ignore_names=['x5', 'x6'])
        new_layer_names = set(self.get_layer_names(pruned_model.layers))
        expected_new_layer_names = {'x0', 'x1', 'x5', 'x6', 'x7'}
        self.assertSetEqual(expected_new_layer_names, new_layer_names)

        seq_branches = GP_ModelParser._get_seq_branches(pruned_model)
        expected_seq_branches_names = [['x0', 'x1'], ['x1', 'x5', 'x6'], ['x6', 'x7'], ['x1', 'x6']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)
        # grow
        new_input_layer = pruned_model.layers[0]
        x8_l = tf.keras.layers.Dense(10, name='x8')
        x9_l = tf.keras.layers.Dense(10, name='x9')
        x10_l = tf.keras.layers.Add(name='x10')
        x8 = x8_l(new_input_layer.output)
        x9 = x9_l(new_input_layer.output)
        x10 = x10_l([x8, x9])
        # MUST NOT be connected!!!
        new_merge_layer = pruned_model.layers[1]
        insert_branch = structure_adaption.InsertBranch([new_input_layer, x8_l, x9_l, x10_l], new_merge_layer)
        inputs = pruned_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [inputs2, outputs2] = structure_adaption.GrowingPruningModel.preprocess_retrace_copy_adapt(input_layers,
                                                                                                   grow_branch_es=insert_branch,
                                                                                                   skip_mismatch=True,
                                                                                                   insert_sequential=True)
        grown_model = tf.keras.Model(name=model.name, inputs=inputs2[0].output, outputs=outputs2[0].output)
        grown_model_weights = grown_model.trainable_variables
        structure_adaption.GrowingPruningModel.maybe_carry_optimizer_slots(optimizer, pruned_model_weights,
                                                                           grown_model_weights, ignore_names=['x1', 'x5', 'x6'])
        optimizer_slots_grown_dict = self.get_optimizer_slot_dict(optimizer)
        self.assertEqualSlotsCarried(optimizer_slots_prun_dict, optimizer_slots_grown_dict,
                                     ignore_names=['x1', 'x5', 'x6'])
        expected_new_layer_names = {'x0', 'x1', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'}
        new_layer_names = set(self.get_layer_names(grown_model.layers))
        self.assertSetEqual(expected_new_layer_names, new_layer_names)
        seq_branches = GP_ModelParser._get_seq_branches(grown_model)
        print_branches(seq_branches)
        expected_seq_branches_names = [['x0', 'x8', 'x10'], ['x10', 'x1'], ['x1', 'x5', 'x6'], ['x6', 'x7'],
                                       ['x1', 'x6'], ['x0', 'x9', 'x10']]
        self.assertEqualSeqBranchesNames(expected_seq_branches_names, seq_branches)

    def testGrowingPruningLayerConnection(self):
        inputs = tf.keras.Input((32, 32, 3), name='x0')
        x1_l = tf.keras.layers.Dense(3, activation='relu', name='x1')
        x1_gpl = GP_ModelParser.parse_layer(x1_l); x1 = x1_gpl(inputs)
        x2_l = tf.keras.layers.Dense(3, activation='relu', name='x2')
        x2_gpl = GP_ModelParser.parse_layer(x2_l); outputs = x2_gpl(x1)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x0_l = model._input_layers[0]
        self.assertIs(x1_gpl, x0_l.outbound_nodes[0].outbound_layer)
        self.assertIs(x0_l, x1_gpl.inbound_nodes[0].inbound_layers)
        self.assertIs(x1_gpl, x2_gpl.inbound_nodes[0].inbound_layers)
        self.assertIs(x2_gpl, x1_gpl.outbound_nodes[0].outbound_layer)


if __name__ == '__main__':
    tf.test.main()

