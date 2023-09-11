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



import queue
import sys
import numpy as np
import tensorflow as tf
import re
import logging
from collections import deque
from bidict import bidict
from abc import ABC

import StructureAdaption.SupportedLayersConfig as SupportedLayersConfig

#logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(levelname)s %(filename)s:%(lineno)s: %(message)s',
#    handlers=[logging.StreamHandler(sys.stdout)])


def filter_weight_name(weight_name):
    """ Removes every substring with a / at the end, so the layer information is removed
    and only the weight name remains"""
    return re.sub(r'[_a-zA-Z0-9]+/', '', weight_name)


def get_change_dim(old_shape, new_shape):
    change_dim = None
    for i in range(len(old_shape)):
        if old_shape[i] != new_shape[i]:
            change_dim = i
    if change_dim is None:
        return None, None
    diff_shape = new_shape.copy()
    diff_shape[change_dim] = new_shape[change_dim] - old_shape[change_dim]
    return change_dim, diff_shape


def carry_optimizer_slots(optimizer, old_variables, new_variables, prun_indices=None):
    """ Only necessary for neuron level pruning, because optimizer will register new slots automatically. """
    if optimizer:
        optimizer._create_slots(new_variables)
        for old_variable, new_variable in zip(old_variables, new_variables):  # Relies on ordering, problematic?
            old_name = filter_weight_name(old_variable.name)
            new_name = filter_weight_name(new_variable.name)
            assert old_name == new_name
            old_shape = old_variable.get_shape().as_list()
            new_shape = new_variable.get_shape().as_list()

            change_dim, diff_shape = get_change_dim(old_shape, new_shape)
            for slot_name in sorted(optimizer.get_slot_names()):
                # Usually slot names only ['momentum']
                old_slot_variable = optimizer.get_slot(old_variable, slot_name)
                new_slot_variable = optimizer.get_slot(new_variable, slot_name)
                old_slot_var_np = old_slot_variable.numpy()
                if change_dim is None:
                    new_slot_var_np = old_slot_var_np
                elif prun_indices:
                    new_slot_var_np = np.delete(old_slot_var_np, prun_indices, axis=change_dim)
                else:
                    # init with zeros
                    add_slot_var_np = np.zeros(tuple(diff_shape))
                    new_slot_var_np = np.concatenate((old_slot_var_np, add_slot_var_np),
                                                     axis=change_dim)
                new_slot_variable.assign(new_slot_var_np)

        remove_optimizer_slots(optimizer, old_variables)


def remove_optimizer_slots(optimizer, variables):
    """ Remove old slots from the optimizer."""
    for old_variable in variables:
        key = (old_variable._shared_name if old_variable._in_graph_mode
               else old_variable._unique_id)
        optimizer._slots.pop(key, None)


class DynamicClassCreator:
    def __init__(self):
        self.created_classes = {}

    def __call__(self, Base):
        name = Base.__name__
        if name in self.created_classes:
            return self.created_classes[name]

        class AdaptionLayer(AdaptionLayerBase, Base):
            def __init__(self, layer_adjust_config, *args, **kwargs):
                super().__init__(layer_adjust_config, *args, **kwargs)

            @classmethod
            def from_config(cls, config):
                layer_adjust_config = config.pop("layer_adjust_config")
                return cls(layer_adjust_config, **config)

            # TODO try later
            #def call(self, inputs, *args, **kwargs):
            #    outputs = self._layer(inputs, *args, **kwargs)
            #    for _, callback_fn in self._callbacks.items():
            #        inputs, outputs = callback_fn(inputs, outputs)
            #    return outputs

        self.created_classes[name] = AdaptionLayer
        return AdaptionLayer


class_creator = DynamicClassCreator()


# This is a mixin class that serves for inheritance for checks and carries all methods that
class AdaptionLayerBase: # reintroduce ABC?
    def __init__(self, layer_adjust_config, *args, importance_measure=None, callbacks=None, **kwargs):
        self.layer_adjust_config = layer_adjust_config
        self.importance_measure = importance_measure
        self.callbacks = callbacks
        super().__init__(*args, **kwargs)

    def get_config(self):
        config_dict = super().get_config()  # Should access Base
        config_dict.update(dict(layer_adjust_config=self.layer_adjust_config,
                                importance_measure=self.importance_measure,
                                callbacks=self.callbacks))
        return config_dict

    def get_units_count(self):
        units_name = self.layer_adjust_config.units_name
        layer_config = self.get_config()
        return layer_config[units_name]


class GrowingPruningTuple:
    instance_count = 0

    def __init__(self, layer_tuple):
        self.name = f'grow_prun_tuple_{GrowingPruningTuple.instance_count}'
        GrowingPruningTuple.instance_count += 1
        self.layer_tuple = layer_tuple
        self.first_layer = layer_tuple[0]
        self.intermediate_layers = layer_tuple[1:-1]
        self.last_layer = layer_tuple[-1]

        # Purpose? Logging: Number of neurons that was in the last step grown or pruned
        self.last_pruned_count = 0
        self.last_grown_count = 0

    def get_trainable_variables_tuple(self):
        """ Used for optimizer slots carry over."""
        trainable_variables = self.first_layer.trainable_variables
        if self.intermediate_layers:
            for layer in self.intermediate_layers:
                trainable_variables = trainable_variables + layer.trainable_variables
        trainable_variables = trainable_variables + self.last_layer.trainable_variables
        return trainable_variables


def add(layer, n_new, init_values, change_kind):
    """ init_values: either a list of np.arrays or a list of initializers for each weight"""
    # add weights
    new_weights = []
    old_weights = layer.weights
    change_dict = getattr(layer.layer_adjust_config, f'change_{change_kind}', None)
    if not change_dict:
        raise ValueError(f'change_kind: {change_kind} is not supported.')
    # Allows for example for use_bias=False
    weights_count = len(old_weights)

    if init_values is None:
        init_values = [None for _ in range(weights_count)]

    for i, old_weight in zip(range(weights_count), old_weights):
        weight_name = filter_weight_name(old_weight.name)
        init_value = init_values[i]
        if weight_name in change_dict:
            add_axis, default_init = get_add_axis_def_init(change_dict[weight_name])
            add_weight = None
            if init_value is None:
                init_value = default_init

            if isinstance(init_value, tf.keras.initializers.Initializer):
                dtype = old_weight.dtype
                add_weight_shape = old_weight.get_shape().as_list()
                add_weight_shape[add_axis] = n_new
                add_weight_tensor = init_value(add_weight_shape, dtype)
                add_weight = add_weight_tensor.numpy()
            elif isinstance(init_value, np.ndarray):
                add_weight = init_value
            elif isinstance(init_value, tf.Tensor):
                add_weight_tensor = init_value
                add_weight = add_weight_tensor.numpy()

            new_weights.append(np.concatenate((old_weight.numpy(), add_weight), axis=add_axis))
        else:
            logging.info(f'Weight name {old_weight.name} (filtered to: {weight_name}) '
                            f'not recognised in {layer.name}.')
            new_weights.append(old_weight.numpy())

    # make new layer
    layer_config = layer.get_config()
    layer_config.update({'weights': new_weights})
    if change_kind == 'units':
        # update number_of_units in layer_config
        units_name = layer.layer_adjust_config.units_name
        new_units_count = layer_config[units_name] + n_new
        layer_config.update({units_name: new_units_count})

    return layer.__class__.from_config(layer_config)


def get_add_axis_def_init(change_object):
    """ If no default initializer is set, this only returns the change axis. Otherwise it returns both."""
    if isinstance(change_object, int):
        return change_object, None
    else:
        return change_object.axis, change_object.def_init


def remove(layer, indices, change_kind):
    """ Removes parts of the weights and scaling_factors according to the
    indices and the change_kind.

    Args:
        indices: Integer or list of integers of indices to be deleted.
        change_kind: One of ['units', 'intermediate', 'inputs']."""
    delete_count = len(indices)
    # delete weights
    new_weights = []
    old_weights = layer.weights
    change_dict = getattr(layer.layer_adjust_config, f'change_{change_kind}', None)
    if not change_dict:
        raise ValueError(f'change_kind: {change_kind} is not supported.')

    for old_weight in old_weights:
        weight_name = filter_weight_name(old_weight.name)
        if weight_name in change_dict:
            del_axis, _ = get_add_axis_def_init(change_dict[weight_name])
            new_weights.append(np.delete(old_weight.numpy(), indices, axis=del_axis))
        else:
            logging.info(f'Weight name {old_weight.name} (filtered to: {weight_name}) '
                         f'not recognised in {layer.name}.')
            new_weights.append(old_weight.numpy())

    # make new layer
    layer_config = layer.get_config()
    layer_config.update({'weights': new_weights})
    if change_kind == 'units':
        # update number_of_units in layer_config
        units_name = layer.layer_adjust_config.units_name
        new_units_count = layer_config[units_name] - delete_count
        layer_config.update({units_name: new_units_count})

    return layer.__class__.from_config(layer_config)


class Branch:
    def __init__(self, layers):
        self.layers = layers

    def __copy__(self):
        copied_branch = Branch(None)
        copied_branch.layers = self.layers.copy()
        return copied_branch

    @property
    def start(self):
        return self.layers[0]

    @start.setter
    def start(self, new_start):
        self.layers[0] = new_start

    @property
    def end(self):
        return self.layers[-1]

    @end.setter
    def end(self, new_end):
        self.layers[-1] = new_end

    @property
    def intermediate_layers(self):
        # These layers are usable for neuron-level growing and pruning
        return self.layers[1:-1]

    @intermediate_layers.setter
    def intermediate_layers(self, new_intermediate_layers):
        self.layers = [self.start] + new_intermediate_layers + [self.end]

    def add_layer(self, layer):
        self.layers.append(layer)

    def __eq__(self, other):
        return self.start is other.start and set(self.intermediate_layers) == set(other.intermediate_layers) \
               and self.end is other.end

    def __hash__(self):
        hash_ = 0
        for layer in self.layers:
            hash_ += hash(layer)
        return hash_


class ParallelBranch(Branch):
    """ Like Branch, but intermediate layers not necessarily in any order.
        Represents a collection of purely parallel branches"""
    def __init__(self, layers, branches):
        super().__init__(layers)
        self.branches = branches


class SequentialParallelBranch(Branch):
    """ Like Branch, but intermediate layers not necessarily in any order.
        Represents a collection of sequential and parallel portions"""
    def __init__(self, layers, branches):
        super().__init__(layers)
        self.branches = branches


class InsertBranch(Branch):
    """ Represents branch that should be inserted into the network."""
    def __init__(self, layers, connection_layer):
        all_layers = layers + [connection_layer]
        super().__init__(all_layers)


class GrowingPruningModelParser:
    def __init__(self, supported_layers_config=None):
        # Setting class attributes for growing and pruning
        self.supported_layers_config = supported_layers_config

    def parse_model_growing_pruning(self, model):
        internal_model = self.parse_model_parallel(model)

        sequential_branches = self._get_seq_branches(internal_model)
        parallel_branches = self._get_par_branches(internal_model, sequential_branches)

        growing_pruning_tuples = self._get_growing_pruning_tuples_par(sequential_branches)

        return internal_model, growing_pruning_tuples, sequential_branches, parallel_branches

    def parse_model_parallel(self, model):
        # TODO: Maybe just accept list of layers as input
        class ParsedPair:
            def __init__(self, layer, parsed_layer):
                self.layer = layer
                self.parsed_layer = parsed_layer
                self.input_argument = False  # Means parsed layer is valid as an input argument

        def get_inbound_layers(layer):
            return layer.inbound_nodes[0].inbound_layers

        layers = model.layers
        parsed_pairs = []
        parsed_pairs_dict = {}
        new_layers = []
        new_inputs = []
        for layer in layers:
            parsed_layer = self.parse_layer(layer)
            new_layers.append(parsed_layer)
            parsed_pair = ParsedPair(layer, parsed_layer)
            inbound_layers = get_inbound_layers(layer)
            if isinstance(inbound_layers, list) and len(inbound_layers) == 0:
                parsed_pair.input_argument = True
                new_inputs.append(parsed_layer.output)
            parsed_pairs.append(parsed_pair)
            parsed_pairs_dict.update({layer: parsed_pair})

        while len(parsed_pairs) != 0:
            parsed_pair = parsed_pairs.pop(0)
            inbound_layers = get_inbound_layers(parsed_pair.layer)
            if isinstance(inbound_layers, list):
                if len(inbound_layers) == 0:
                    continue
                parsed_inbound_layers_output = []
                for inbound_layer in inbound_layers:
                    input_parsed_pair = parsed_pairs_dict[inbound_layer]
                    if not input_parsed_pair.input_argument:
                        # TODO better append left and make deque? yes, but works for now
                        parsed_pairs.append(parsed_pair)
                        continue
                    parsed_inbound_layers_output.append(input_parsed_pair.parsed_layer.output)
                parsed_pair.parsed_layer(parsed_inbound_layers_output)
                parsed_pair.input_argument = True
            else:
                input_parsed_pair = parsed_pairs_dict[inbound_layers]
                if input_parsed_pair.input_argument:
                    parsed_pair.parsed_layer(input_parsed_pair.parsed_layer.output)
                    parsed_pair.input_argument = True
                else:
                    parsed_pairs.append(parsed_pair)

        new_outputs = []
        unparsed_output_layers = model._output_layers
        for unparsed_output_layer in unparsed_output_layers:
            parsed_pair = parsed_pairs_dict[unparsed_output_layer]
            new_outputs.append(parsed_pair.parsed_layer.output)

        internal_model = tf.keras.Model(name=model.name, inputs=new_inputs, outputs=new_outputs)
        return internal_model

    @staticmethod
    def _get_seq_branches(model):
        # works
        # depth first search
        # assumes one input and one output

        visited_layers = []
        open_branches = deque()
        complete_branches = []

        def check_outbound_layers(layer):
            outbound_layers = [out_node.outbound_layer for out_node in layer.outbound_nodes]
            if isinstance(outbound_layers, list) and len(outbound_layers) > 1:
                return True
            return False

        def get_outbound_layers(layer):
            outbound_layers = [out_node.outbound_layer for out_node in layer.outbound_nodes]
            return outbound_layers

        def check_inbound_layers(layer, output):
            inbound_layers = layer.inbound_nodes[0].inbound_layers
            if isinstance(inbound_layers, list) and len(inbound_layers) > 1 or layer is output:
                return True
            return False

        input = model._input_layers[0]
        output = model._output_layers[0]
        current_node = input
        current_branch = Branch([input])

        while current_branch:
            old_current_node = current_node
            logging.debug(f'current node: {current_node.name}')
            if check_inbound_layers(current_node, output):
                # current_node is merging node
                complete_branches.append(current_branch)

                if current_node not in visited_layers and current_node is not output:
                    # current_node not visited yet
                    # TODO: can be unified with list in both cases
                    if check_outbound_layers(current_node):
                        # current_node has multiple outputs
                        current_branch = Branch([current_node])
                        current_branch.add_layer(get_outbound_layers(current_node)[0])
                        for other_layer in get_outbound_layers(current_node)[1:]:
                            other_branch = Branch([current_node])
                            other_branch.add_layer(other_layer)
                            open_branches.append(other_branch)
                    else:
                        # current_node has one output
                        current_branch = Branch([current_node])
                        current_branch.add_layer(get_outbound_layers(current_node)[0])
                    visited_layers.append(old_current_node)
                    current_node = current_branch.layers[-1]
                    continue
                else:
                    if len(open_branches):
                        current_branch = open_branches.pop()
                        current_node = current_branch.layers[-1]
                    else:
                        current_branch = None
                    visited_layers.append(old_current_node)
                    continue

            elif check_outbound_layers(current_node):
                # current node branches out
                if current_node not in visited_layers:
                    # If starting branch is only length == 1, throw it away
                    if len(current_branch.layers) > 1:
                        complete_branches.append(current_branch)

                    current_branch = Branch([current_node])
                    current_branch.add_layer(get_outbound_layers(current_node)[0])
                    for other_layer in get_outbound_layers(current_node)[1:]:
                        new_other_branch = Branch([current_node])
                        new_other_branch.add_layer(other_layer)
                        open_branches.append(new_other_branch)

                    current_node = current_branch.layers[-1]
                    visited_layers.append(old_current_node)
                    continue
                else:
                    if len(open_branches):
                        current_branch = open_branches.pop()
                        current_node = current_branch.layers[-1]
                    else:
                        current_branch = None
                    visited_layers.append(old_current_node)
                    continue
            else:
                # current node neither merges or branches
                next_node = get_outbound_layers(current_node)[0]
                current_branch.add_layer(next_node)
                current_node = next_node
                visited_layers.append(old_current_node)
                continue

        return complete_branches

    @staticmethod
    def _get_par_branches(model, sequential_branches):
        # bad time complexity, but it works for now
        if GrowingPruningModelParser._check_sequential_model(model):
            return []
        input = model._input_layers[0]
        output = model._output_layers[0]

        def check_sequentiality_start(branch):
            start_layer = branch.start
            outbound_layers = [out_node.outbound_layer for out_node in start_layer.outbound_nodes]
            for outbound_layer in outbound_layers:
                # relies on 'in' operator to use identity and not equality
                if outbound_layer not in branch.layers:
                    return False
            return True

        def check_sequentiality_end(branch):
            end_layer = branch.end
            inbound_layers = end_layer.inbound_nodes[0].inbound_layers
            if not isinstance(inbound_layers, list):
                return True
            for inbound_layer in inbound_layers:
                # relies on 'in' operator to use identity and not equality
                if inbound_layer not in branch.layers:
                    return False
            return True

        def check_start_input(branch):
            start_layer = branch.start
            return start_layer is input

        def check_end_output(branch):
            end_layer = branch.end
            return end_layer is output

        def merge_parallel(parallel_branches):
            new_start = parallel_branches[0].start
            new_end = parallel_branches[0].end
            new_intermediate_layers = set()
            for branch in parallel_branches:
                new_intermediate_layers = new_intermediate_layers.union(set(branch.intermediate_layers))
            new_intermediate_layers = list(new_intermediate_layers)
            new_layers = [new_start] + new_intermediate_layers + [new_end]
            return ParallelBranch(new_layers, parallel_branches)

        def merge_sequential(low, high):
            if isinstance(high, SequentialParallelBranch) and not isinstance(low, SequentialParallelBranch):
                layers = [low.start] + low.intermediate_layers + high.layers
                branches = [low] + high.branches
                return SequentialParallelBranch(layers, branches)
            elif not isinstance(high, SequentialParallelBranch) and isinstance(low, SequentialParallelBranch):
                layers = [low.start] + low.intermediate_layers + high.layers
                branches = low.branches + [high]
                return SequentialParallelBranch(layers, branches)
            elif isinstance(high, SequentialParallelBranch) and isinstance(low, SequentialParallelBranch):
                layers = [low.start] + low.intermediate_layers + high.layers
                branches = low.branches + high.branches
                return SequentialParallelBranch(layers, branches)
            else:
                layers = low.layers + high.intermediate_layers + [high.end]
                branches = [low, high]
                return SequentialParallelBranch(layers, branches)

        # All sequential branches
        seq_par_branches = sequential_branches.copy()
        # New sequential and/or parallel branches that need to be checked for parallel branches
        new_seq_par_branches = deque(sequential_branches.copy())
        # All parallel branches
        par_branches = []
        # New parallel branches that need to be checked for sequential additions
        new_par_branches = deque()
        complete = False
        while not complete:
            # find all parallel branches, could also check all merging nodes
            i = 0
            max = len(new_seq_par_branches)
            while i < max:
                parallel_branches = []
                current_branch = new_seq_par_branches.pop()
                for other_branch in seq_par_branches:
                    same_start_end = current_branch.start is other_branch.start and \
                                     current_branch.end is other_branch.end
                    is_same_branch = current_branch is other_branch
                    if same_start_end and not is_same_branch:
                        parallel_branches.append(other_branch)
                if parallel_branches:
                    parallel_branches.append(current_branch)
                    new_par_branch = merge_parallel(parallel_branches)
                    # check that all parallel branches are included
                    if check_sequentiality_start(new_par_branch) or check_sequentiality_end(new_par_branch):
                        par_branches.append(new_par_branch)
                        new_par_branches.append(new_par_branch)
                        seq_par_branches.append(new_par_branch)
                        for branch in parallel_branches:
                            if branch in new_seq_par_branches:
                                new_seq_par_branches.remove(branch)
                                i += 1
                            seq_par_branches.remove(branch)
                        logging.debug(
                            f'found new parallel branch from: {new_par_branch.start.name} to {new_par_branch.end.name}')
                    else:
                        new_seq_par_branches.appendleft(current_branch)
                i += 1

            # connect sequential branches sequentially
            i = 0
            max = len(new_par_branches)
            while i < max:
                current_branch = new_par_branches.pop()
                current_branch_finished = True
                current_branch_changed = False
                # attach as many sequential branches as possible
                # at start
                if check_sequentiality_start(current_branch) and not check_start_input(current_branch):
                    current_branch_finished = False
                    for add_branch in seq_par_branches:
                        if current_branch.start is add_branch.end and check_sequentiality_end(add_branch):
                            new_current_branch = merge_sequential(add_branch, current_branch)
                            if add_branch in new_par_branches:
                                new_par_branches.remove(add_branch)
                                i += 1
                            if add_branch in seq_par_branches:
                                seq_par_branches.remove(add_branch)
                            if current_branch in new_par_branches:
                                new_par_branches.remove(current_branch)
                            if current_branch in seq_par_branches:
                                seq_par_branches.remove(current_branch)
                            current_branch = new_current_branch
                            current_branch_changed = True
                            break
                # at end
                if check_sequentiality_end(current_branch) and not check_end_output(current_branch):
                    current_branch_finished = False
                    for add_branch in seq_par_branches:
                        if current_branch.end is add_branch.start and check_sequentiality_start(add_branch):
                            new_current_branch = merge_sequential(current_branch, add_branch)
                            if add_branch in new_par_branches:
                                new_par_branches.remove(add_branch)
                                i += 1
                            if add_branch in seq_par_branches:
                                seq_par_branches.remove(add_branch)
                            if current_branch in new_par_branches:
                                new_par_branches.remove(current_branch)
                            if current_branch in seq_par_branches:
                                seq_par_branches.remove(current_branch)
                            current_branch = new_current_branch
                            current_branch_changed = True
                            break
                i += 1

                if not current_branch_finished:
                    if current_branch_changed:
                        seq_par_branches.append(current_branch)
                    new_par_branches.appendleft(current_branch)
                else:
                    new_seq_par_branches.append(current_branch)
                    seq_par_branches.append(current_branch)
                    # What if other branches for completion not ready yet?
                    logging.debug(f'added seq par branch from: {current_branch.start.name} to {current_branch.end.name}')
                if current_branch.end is output and current_branch.start is input:
                    complete = True

        return par_branches

    @staticmethod
    def _get_growing_pruning_tuples_par(seq_branches):
        growing_pruning_tuples = []
        current_tuple = []
        for branch in seq_branches:
            layers = branch.intermediate_layers
            for i in range(len(layers) - 1):
                if not isinstance(layers[i], AdaptionLayerBase):
                    continue
                is_grow_prunable = 'grow_prun_able' in layers[i].layer_adjust_config.tags
                if is_grow_prunable:
                    current_tuple.append(layers[i])
                    for j in range(i + 1, len(layers)):
                        if not isinstance(layers[j], AdaptionLayerBase):
                            current_tuple = []
                            break
                        is_grow_prunable = 'grow_prun_able' in layers[j].layer_adjust_config.tags
                        is_intermediate = 'intermediate' in layers[j].layer_adjust_config.tags

                        if is_intermediate:
                            current_tuple.append(layers[j])
                        elif is_grow_prunable:
                            current_tuple.append(layers[j])
                            tuple = GrowingPruningTuple(current_tuple)
                            growing_pruning_tuples.append(tuple)
                            current_tuple = []
                            break
        return growing_pruning_tuples

    def parse_layer(self, layer):
        layer_config = layer.get_config()
        # Carry over weights
        if layer.built and layer.weights:
            layer_config.update(dict(weights=layer.get_weights()))
        cls_name = type(layer).__name__
        if hasattr(self.supported_layers_config, cls_name):
            layer_adjust_config = getattr(self.supported_layers_config, cls_name)
            deep_layer_copy = class_creator(type(layer))(layer_adjust_config, **layer_config)
            logging.debug('Registered supported layer: %s', layer.name)
            return deep_layer_copy
        else:
            logging.debug('Registered not supported layer: %s', layer_config['name'])
            deep_layer_copy = type(layer).from_config(layer_config)
            return deep_layer_copy

    @staticmethod
    def _check_sequential_model(model):
        for layer in model.layers:
            if isinstance(layer.input, list) and len(layer.input) != 1:
                return False
        return True


def parse_model(input_model):
    Parser = GrowingPruningModelParser(SupportedLayersConfig.get_config())
    [internal_model, gp_tuples, seq_branches, par_branches] = Parser.parse_model_growing_pruning(input_model)
    return GrowingPruningModel(internal_model, gp_tuples, seq_branches, par_branches)


class GrowingPruningModel:
    """ Handler for the internal model, GrowPrunTuples and sequential and parallel branches. """
    def __init__(self, internal_model, grow_prun_tuples, seq_branches, par_branches):
        self.internal_model = internal_model
        self.grow_prun_tuples = grow_prun_tuples
        self.sequential_branches = seq_branches
        self.parallel_branches = par_branches

    def grow(self, gp_tuple, n_new, init_dict, optimizer, compile_fn):
        old_variables = gp_tuple.get_trainable_variables_tuple()
        init_first = init_dict['first']
        init_last = init_dict['last']
        inits_intermediate = None

        pre_cloned_dict = {}
        if 'intermediate' in init_dict:
            inits_intermediate = init_dict['intermediate']
        new_first_layer = add(gp_tuple.first_layer, n_new=n_new, init_values=init_first, change_kind='units')
        pre_cloned_dict.update({gp_tuple.first_layer: new_first_layer})
        gp_tuple.first_layer = new_first_layer
        if gp_tuple.intermediate_layers:
            layers_count = len(gp_tuple.intermediate_layers)
            inits_count = len(inits_intermediate)
            new_intermediate_layers = []
            if layers_count != inits_count:
                raise ValueError(f'number of intermediate layers ({layers_count}) doesn\'t match number of '
                                 f'intermediate layer initializers ({inits_count}).')
            for layer, init in zip(gp_tuple.intermediate_layers, inits_intermediate):
                new_intermediate_layer = add(layer, n_new=n_new, init_values=init, change_kind='intermediate')
                new_intermediate_layers.append(new_intermediate_layer)
                pre_cloned_dict.update({layer: new_intermediate_layer})
            gp_tuple.intermediate_layers = new_intermediate_layers
        new_last_layer = add(gp_tuple.last_layer, n_new=n_new, init_values=init_last, change_kind='inputs')
        pre_cloned_dict.update({gp_tuple.last_layer: new_last_layer})
        gp_tuple.last_layer = new_last_layer
        self.last_grown_count = n_new

        self.re_init_out_retraced(pre_cloned_dict, compile_fn)

        new_variables = gp_tuple.get_trainable_variables_tuple()
        if optimizer:
            carry_optimizer_slots(optimizer, old_variables, new_variables)

    def prun(self, gp_tuple, indices, optimizer, compile_fn):
        old_variables = gp_tuple.get_trainable_variables_tuple()
        pre_cloned_dict = {}

        new_first_layer = remove(gp_tuple.first_layer, indices=indices, change_kind='units')
        pre_cloned_dict.update({gp_tuple.first_layer: new_first_layer})
        gp_tuple.first_layer = new_first_layer

        if gp_tuple.intermediate_layers:
            new_intermediate_layers = []
            for layer in gp_tuple.intermediate_layers:
                new_intermediate_layer = remove(layer, indices=indices, change_kind='intermediate')
                pre_cloned_dict.update({layer: new_intermediate_layer})
                new_intermediate_layers.append(new_intermediate_layer)
            gp_tuple.intermediate_layers = new_intermediate_layers
        new_last_layer = remove(gp_tuple.last_layer, indices=indices, change_kind='inputs')
        pre_cloned_dict.update({gp_tuple.last_layer: new_last_layer})
        gp_tuple.last_layer = new_last_layer
        self.last_pruned_count = len(indices)

        self.re_init_out_retraced(pre_cloned_dict, compile_fn)

        new_variables = gp_tuple.get_trainable_variables_tuple()
        if optimizer:
            carry_optimizer_slots(optimizer, old_variables, new_variables, prun_indices=indices)

    def re_init_out_retraced(self, pre_cloned_layers_dict, compile_fn):
        inputs = self.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [input_layer, output_layer] = self.preprocess_retrace_copy_adapt(input_layers, pre_cloned_layers=pre_cloned_layers_dict)
        # TODO for multiple inputs
        self.internal_model = tf.keras.Model(inputs=input_layer[0].output, outputs=output_layer[0].output, name=self.internal_model.name)
        compile_fn()

        self.sequential_branches = GrowingPruningModelParser._get_seq_branches(self.internal_model)
        self.parallel_branches = GrowingPruningModelParser._get_par_branches(self.internal_model,
                                                                             self.sequential_branches)
        self.grow_prun_tuples = GrowingPruningModelParser._get_growing_pruning_tuples_par(
            self.sequential_branches)

    @staticmethod
    def maybe_carry_optimizer_slots(optimizer, old_variables, new_variables, ignore_names=None):
        if optimizer:
            """ ignore_names ignores weights of layers with the corresponding name."""
            if ignore_names is None:
                ignore_names = []
            optimizer._create_slots(new_variables)
            for old_variable in old_variables:
                old_name = old_variable.name
                for new_variable in new_variables:
                    new_name = new_variable.name
                    for ignore_name in ignore_names:
                        if ignore_name in new_name:
                            break
                    else:
                        if old_name == new_name:
                            for slot_name in sorted(optimizer.get_slot_names()):
                                # Usually slot names only ['momentum']
                                old_slot_var = optimizer.get_slot(old_variable, slot_name)
                                new_slot_var = optimizer.get_slot(new_variable, slot_name)
                                new_slot_var.assign(old_slot_var)

            remove_optimizer_slots(optimizer, old_variables)

    def prun_branch(self, prun_branch, optimizer, compile_fn, carry_optimizer=False, skip_mismatch=False,
                                      leave_residual=False):
        inputs = self.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [input_layer, output_layer] = self.preprocess_retrace_copy_adapt(input_layers, prun_branch_es=prun_branch,
                                                                         skip_mismatch=skip_mismatch,
                                                                         leave_residual=leave_residual)
        old_internal_model = self.internal_model
        self.internal_model = tf.keras.Model(inputs=input_layer[0].output, outputs=output_layer[0].output, name=self.internal_model.name)
        compile_fn()
        old_variables = old_internal_model.trainable_variables
        new_variables = self.internal_model.trainable_variables
        if carry_optimizer:
            self.maybe_carry_optimizer_slots(optimizer, old_variables, new_variables,
                                             ignore_names=[prun_branch.end.name])

        self.sequential_branches = GrowingPruningModelParser._get_seq_branches(self.internal_model)
        self.parallel_branches = GrowingPruningModelParser._get_par_branches(self.internal_model,
                                                                             self.sequential_branches)
        self.grow_prun_tuples = GrowingPruningModelParser._get_growing_pruning_tuples_par(
            self.sequential_branches)

    def grow_branch(self, grow_branch, optimizer, compile_fn, carry_optimizer=False, skip_mismatch=False,
                    insert_sequential=False):
        inputs = self.internal_model.inputs
        input_layers = [input.node.layer for input in inputs]
        [input_layer, output_layer] = self.preprocess_retrace_copy_adapt(input_layers, grow_branch_es=grow_branch,
                                                                         skip_mismatch=skip_mismatch,
                                                                         insert_sequential=insert_sequential)
        old_internal_model = self.internal_model
        self.internal_model = tf.keras.Model(inputs=input_layer[0].output, outputs=output_layer[0].output,
                                             name=self.internal_model.name)
        compile_fn()
        old_variables = old_internal_model.trainable_variables
        new_variables = self.internal_model.trainable_variables
        if carry_optimizer:
            self.maybe_carry_optimizer_slots(optimizer, old_variables, new_variables)

        self.sequential_branches = GrowingPruningModelParser._get_seq_branches(self.internal_model)
        self.parallel_branches = GrowingPruningModelParser._get_par_branches(self.internal_model,
                                                                             self.sequential_branches)
        self.grow_prun_tuples = GrowingPruningModelParser._get_growing_pruning_tuples_par(
            self.sequential_branches)

    def summary(self):
        self.internal_model.summary()

    def __call__(self, input, *args, **kwargs):
        return self.internal_model(input)

    class LayerPair:
        def __init__(self, first, second):
            self.first = first
            self.second = second

        def __hash__(self):
            return 3 * hash(self.first) + hash(self.second)

    @staticmethod
    def get_edges_pruning(branch):
        # Needed only for removing
        # TODO: Problem is this is not a multiset. Duplicates get deleted
        edges = set()
        parsed = set()
        start = branch.start
        end = branch.end
        outer_nodes = queue.SimpleQueue()
        current_layer = None
        if not branch.intermediate_layers:
            layer_pair = GrowingPruningModel.LayerPair(start, end)
            edges.add(layer_pair)
            return edges
        # first line:
        for outbound_layer in branch.intermediate_layers + [branch.end]:
            if start == outbound_layer.inbound_nodes[0].inbound_layers:
                outer_nodes.put_nowait(outbound_layer)
                layer_pair = GrowingPruningModel.LayerPair(start, outbound_layer)
                edges.add(layer_pair)

        while not outer_nodes.empty():
            current_layer = outer_nodes.get_nowait()
            outbound_layers = GrowingPruningModel.get_outbound_layers(current_layer)
            if current_layer == end:
                continue
            for outbound_layer in outbound_layers:
                layer_pair = GrowingPruningModel.LayerPair(current_layer, outbound_layer)
                edges.add(layer_pair)
                if outbound_layer not in parsed:
                    parsed.add(outbound_layer)
                    outer_nodes.put_nowait(outbound_layer)
        return edges

    @staticmethod
    def get_edges_growing(branch):
        pre_cloned_layers = dict()
        end_edges = bidict()
        start = branch.start
        end = branch.end
        if not branch.intermediate_layers:
            end_edges.update({end: start})
        for layer in [branch.start] + branch.intermediate_layers:
            if not layer.outbound_nodes:
                end_edges.update({end: layer})

        Parser = GrowingPruningModelParser(SupportedLayersConfig.get_config())
        for layer in branch.intermediate_layers:
            if not isinstance(layer, AdaptionLayerBase):
                cloned_layer = Parser.parse_layer(layer)
                pre_cloned_layers.update({layer: cloned_layer})

        return end_edges, pre_cloned_layers

    @staticmethod
    def get_inbound_layers(layer):
        inbound_layers = layer.inbound_nodes[0].inbound_layers
        if isinstance(inbound_layers, list):
            return inbound_layers
        else:
            return [inbound_layers]

    @staticmethod
    def get_outbound_layers(layer):
        outbound_layers = [out_node.outbound_layer for out_node in layer.outbound_nodes]
        if isinstance(outbound_layers, list):
            return outbound_layers
        else:
            return [outbound_layers]

    @staticmethod
    def retrace_copy_adapt(input_layers, pre_cloned_layers=None, pre_cloned_and_called_layers=None,
                           remove_edges_set=None, add_edges_ends=None, skip_mismatch=False):
        # TODO: Add edges ends duplicates not possible
        # Allows for multiple pruning and growing branches at once
        if remove_edges_set:
            remove_edges_hash_set = [edge.__hash__() for edge in remove_edges_set]
        else:
            remove_edges_hash_set = []
        new_merge_nodes_count = 0

        def clone_layer(layer):
            if pre_cloned_layers and layer in pre_cloned_layers:
                return pre_cloned_layers[layer]
            layer_config = layer.get_config()
            if not skip_mismatch and layer.weights:
                layer_config.update(dict(weights=layer.weights))
            return layer.__class__.from_config(layer_config)

        def maybe_set_weights(cloned_layer, outbound_layer):
            if cloned_layer.weights:
                old_weights = outbound_layer.weights
                new_weights = cloned_layer.weights
                weights_match = True
                for old_weight, new_weight in zip(old_weights, new_weights):
                    if old_weight.shape != new_weight.shape:
                        logging.warning("assigned weights shape mismatch for layer %s", outbound_layer.name)
                        weights_match = False
                if weights_match:
                    cloned_layer.set_weights(old_weights)

        def maybe_make_incomplete_inputs(inbound_layers_count, outbound_layer):
            if outbound_layer in incomplete_inputs:
                incomplete_inputs_list = incomplete_inputs[outbound_layer]
            else:
                incomplete_inputs_list = IncompleteInputs(inbound_layers_count)
                incomplete_inputs.update({outbound_layer: incomplete_inputs_list})
            return incomplete_inputs_list

        def clone_and_connect(outbound_layer=None, inputs=None):
            cloned_layer = clone_layer(outbound_layer)
            cloned_layer(inputs)
            cloned_and_called.update({outbound_layer: cloned_layer})
            outer_nodes.append(outbound_layer)  # put_nowait(outbound_layer)
            return cloned_layer

        def connect_full_incomplete_input(incomplete_inputs_list, outbound_layer, inbound_layers_count):
            logging.debug("connecting full incomplete input for layer %s", outbound_layer.name)
            if incomplete_inputs_list.size == 1:
                if inbound_layers_count == 1:
                    # no need to remove a merge node
                    connecting_inputs = cloned_and_called[incomplete_inputs_list.inputs[0]].output
                else:
                    logging.debug("removing merge node: %s", outbound_layer.name)
                    # Other inputs have been pruned
                    new_current_layer = incomplete_inputs_list.inputs[0]
                    new_outbound_layers = GrowingPruningModel.get_outbound_layers(outbound_layer)
                    for new_outbound_layer in new_outbound_layers:
                        gather_imputs_clone_and_connect(new_outbound_layer, new_current_layer)
                    return None
            elif incomplete_inputs_list.size > 1 and not isinstance(outbound_layer.inbound_nodes[0].inbound_layers, list):
                # introduce new merge node
                logging.debug("adding new add node for layer: %s", outbound_layer.name)
                incomplete_inputs = [cloned_and_called[current_input_layer].output for current_input_layer in
                                     incomplete_inputs_list.inputs]
                nonlocal new_merge_nodes_count
                new_merge_node = tf.keras.layers.Add(name='new_add_node' + str(new_merge_nodes_count))
                new_merge_nodes_count += 1
                connecting_inputs = new_merge_node(incomplete_inputs)
            else:
                # clone and connect because list is full
                connecting_inputs = [cloned_and_called[current_input_layer].output for current_input_layer in
                                     incomplete_inputs_list.inputs]
            clone_and_connect(outbound_layer=outbound_layer, inputs=connecting_inputs)

        def gather_imputs_clone_and_connect(outbound_layer, current_layer):
            """ outbound_layer: layer to be cloned
            layer: one input of layer to be cloned
            """
            layer_pair = GrowingPruningModel.LayerPair(current_layer, outbound_layer)
            layer_pair_hash = layer_pair.__hash__()
            inbound_layers = GrowingPruningModel.get_inbound_layers(outbound_layer)
            inbound_layers_count = len(inbound_layers)
            cloned_layer = None

            is_add_edges = False
            if add_edges_ends and outbound_layer in add_edges_ends:
                is_add_edges = True

            if is_add_edges and layer_pair_hash in remove_edges_hash_set:
                new_current_layer = add_edges_ends[outbound_layer]
                logging.debug("new add edge layer detected: %s and remove edge detected from %s to %s",
                              new_current_layer.name, current_layer.name, outbound_layer.name)
                incomplete_inputs_list = maybe_make_incomplete_inputs(inbound_layers_count, outbound_layer)
                remove_edges_hash_set.remove(layer_pair_hash)
                incomplete_inputs_list.size -= 1

                # Needed for residual connections
                if new_current_layer in cloned_and_called:
                    incomplete_inputs_list.size += 1
                    incomplete_inputs_list.add(new_current_layer)
                    if new_current_layer in outer_nodes:
                        outer_nodes.remove(new_current_layer)
                    del add_edges_ends[outbound_layer]

                    if incomplete_inputs_list.full():
                        connect_full_incomplete_input(incomplete_inputs_list, outbound_layer, inbound_layers_count)

            elif layer_pair_hash in remove_edges_hash_set:
                logging.debug("remove edge detected from %s to %s", current_layer.name, outbound_layer.name)
                if inbound_layers_count > 1:
                    if outbound_layer in incomplete_inputs:
                        incomplete_inputs_list = incomplete_inputs[outbound_layer]
                        incomplete_inputs_list.size -= 1
                        # No need to check is_add_edges, because it was checked above already
                        if incomplete_inputs_list.size == 0:
                            outer_nodes.append(outbound_layer)
                        elif incomplete_inputs_list.full():
                            connect_full_incomplete_input(incomplete_inputs_list, outbound_layer, inbound_layers_count)
                    else:
                        incomplete_inputs_list = IncompleteInputs(inbound_layers_count-1)
                        incomplete_inputs.update({outbound_layer: incomplete_inputs_list})
                remove_edges_hash_set.remove(layer_pair_hash)
                if inbound_layers_count == 1:
                    # so outbound is parsed also
                    outer_nodes.append(outbound_layer) #put_nowait(outbound_layer)
            else:
                logging.debug("normal edge detected from %s to %s", current_layer.name, outbound_layer.name)
                if inbound_layers_count == 1 and not is_add_edges:
                    logging.debug("connecting: %s to %s", current_layer.name, outbound_layer.name)
                    cloned_layer = clone_and_connect(outbound_layer=outbound_layer, inputs=cloned_and_called[current_layer].output)
                else:
                    incomplete_inputs_list = maybe_make_incomplete_inputs(inbound_layers_count, outbound_layer)
                    incomplete_inputs_list.add(current_layer)
                    logging.debug("adding node %s to incomplete inputs for %s", current_layer.name, outbound_layer.name)
                    # maybe add_edges
                    if is_add_edges:
                        new_current_layer = add_edges_ends[outbound_layer]
                        if new_current_layer in cloned_and_called:
                            logging.debug("new add edge layer detected: %s", new_current_layer.name)
                            del add_edges_ends[outbound_layer]

                            if new_current_layer in outer_nodes:
                                outer_nodes.remove(new_current_layer)

                            incomplete_inputs_list.size += 1
                            incomplete_inputs_list.add(new_current_layer)
                            is_add_edges = False

                    if incomplete_inputs_list.full() and not is_add_edges:
                        connect_full_incomplete_input(incomplete_inputs_list, outbound_layer, inbound_layers_count)

            if cloned_layer and skip_mismatch:
                maybe_set_weights(cloned_layer, outbound_layer)

        class IncompleteInputs:
            def __init__(self, size):
                self.size = size
                self.count = 0
                self.inputs = []

            def add(self, item):
                if self.size == self.count:
                    raise IndexError('Input list is already full')
                self.inputs.append(item)
                self.count += 1

            def full(self):
                return self.size == self.count

        cloned_and_called = dict()
        incomplete_inputs = dict()
        outer_nodes = list() # queue.SimpleQueue()

        cloned_inputs = []
        cloned_outputs = []
        for input_layer in input_layers:
            cloned_layer = clone_layer(input_layer)
            outer_nodes.append(input_layer) #put_nowait(input_layer)
            cloned_and_called.update({input_layer: cloned_layer})
            cloned_inputs.append(cloned_layer)

        #if pre_cloned_and_called_layers:
        #    # Pre cloned and called layers are considered as additional inputs
        #    for pre_cl_ca_layer in pre_cloned_and_called_layers:
        #        outer_nodes.put_nowait(pre_cl_ca_layer)
        #        cloned_layer = pre_cloned_and_called_layers[pre_cl_ca_layer]
        #        cloned_and_called.update({pre_cl_ca_layer: cloned_layer})
        #        # TODO: not all of them are inputs!!!
        #        cloned_inputs.append(cloned_layer)

        while len(outer_nodes): #not outer_nodes.empty():
            current_layer = outer_nodes.pop(0) #outer_nodes.get_nowait()
            logging.debug("outer_nodes: %s", str([outer_node.name for outer_node in outer_nodes]))
            logging.debug("current_layer: %s", current_layer.name)
            outbound_layers = GrowingPruningModel.get_outbound_layers(current_layer)
            logging.debug("outbound layers: %s", str([outbound_layer.name for outbound_layer in outbound_layers]))
            outbound_layers_count = len(outbound_layers)

            if outbound_layers_count == 0:
                logging.debug("found potential new output: %s", current_layer.name)
                if not add_edges_ends or add_edges_ends and current_layer not in add_edges_ends.values():
                    logging.debug("found new output: %s", current_layer.name)
                    cloned_outputs.append(cloned_and_called[current_layer])
                else:
                    # Must be loose end of insert branch
                    new_outbound_layer = add_edges_ends.inverse[current_layer]
                    incomplete_inputs_list = incomplete_inputs[new_outbound_layer]
                    incomplete_inputs_list.size += 1
                    incomplete_inputs_list.add(current_layer)
                    del add_edges_ends[new_outbound_layer]
                    inbound_layers_count = len(GrowingPruningModel.get_inbound_layers(new_outbound_layer))
                    if incomplete_inputs_list.full():
                        connect_full_incomplete_input(incomplete_inputs_list, new_outbound_layer, inbound_layers_count)
            else:
                for outbound_layer in outbound_layers:
                    logging.debug("clone and connect: current layer %s and outbound layer %s", current_layer.name,
                                  outbound_layer.name)
                    gather_imputs_clone_and_connect(outbound_layer, current_layer)

        return cloned_inputs, cloned_outputs

    @staticmethod
    def preprocess_retrace_copy_adapt(input_layers, pre_cloned_layers=None, pre_cloned_and_called_layers=None,
                                      prun_branch_es=None, grow_branch_es=None, skip_mismatch=False,
                                      leave_residual=False, insert_sequential=False):
        add_edges_ends = bidict()
        remove_edges = set()

        if grow_branch_es:
            if isinstance(grow_branch_es, list):
                pre_cloned_layers = dict()
                for grow_branch in grow_branch_es:
                    if insert_sequential:
                        layer_pair = GrowingPruningModel.LayerPair(grow_branch.start, grow_branch.end)
                        remove_edges.add(layer_pair)
                    end_edges, pre_cloned_layers_updt = GrowingPruningModel.get_edges_growing(grow_branch)
                    add_edges_ends.update(end_edges)
                    pre_cloned_layers.update(pre_cloned_layers_updt)
            elif grow_branch_es:
                if insert_sequential:
                    layer_pair = GrowingPruningModel.LayerPair(grow_branch_es.start, grow_branch_es.end)
                    remove_edges.add(layer_pair)
                end_edges, pre_cloned_layers = GrowingPruningModel.get_edges_growing(grow_branch_es)
                add_edges_ends.update(end_edges)

        if prun_branch_es:
            if isinstance(prun_branch_es, list):
                for prun_branch in prun_branch_es:
                    remove_edge = GrowingPruningModel.get_edges_pruning(prun_branch)
                    remove_edges = remove_edges.union(remove_edge)
                    if leave_residual:
                        add_edges_ends.update({prun_branch.end: prun_branch.start})
            elif prun_branch_es:
                remove_edges = GrowingPruningModel.get_edges_pruning(prun_branch_es)
                if leave_residual:
                    add_edges_ends.update({prun_branch_es.end: prun_branch_es.start})

        if remove_edges:
            for edge in remove_edges:
                logging.debug("remove edge found: from %s to %s", edge.first.name, edge.second.name)
        else:
            remove_edges = None
        if add_edges_ends:
            for item in add_edges_ends.items():
                logging.debug("add edge found: from: %s to %s", item[1].name, item[0].name)
        else:
            add_edges_ends = None

        return GrowingPruningModel.retrace_copy_adapt(input_layers,
                                                      pre_cloned_layers=pre_cloned_layers,
                                                      remove_edges_set=remove_edges, add_edges_ends=add_edges_ends,
                                                      skip_mismatch=skip_mismatch)