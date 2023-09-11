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


def sequential_short():
    inputs = tf.keras.Input(shape=[10], dtype=tf.dtypes.float32, name='x0')
    x1 = tf.keras.layers.Dense(units=16, activation='relu', name='x1')(inputs)
    x2 = tf.keras.layers.Dense(units=16, activation='relu', name='x2')(x1)
    outputs = tf.keras.layers.Softmax(name='x3')(x2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def sequential_long():
    inputs = tf.keras.Input(shape=[10], dtype=tf.dtypes.float32, name='x0')
    x1 = tf.keras.layers.Dense(units=16, activation='relu', name='x1')(inputs)
    x2 = tf.keras.layers.Dense(units=16, activation='relu', name='x2')(x1)
    x3 = tf.keras.layers.Dense(units=16, activation='relu', name='x3')(x2)
    x4 = tf.keras.layers.Dense(units=16, activation='relu', name='x4')(x3)
    outputs = tf.keras.layers.Softmax(name='x5')(x4)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def non_sequential_simple():
    inputs = tf.keras.Input(shape=[10], dtype=tf.dtypes.float32, name='x0')
    x1 = tf.keras.layers.Dense(units=16, activation='relu', name='x1')(inputs)
    x2 = tf.keras.layers.Dense(units=16, activation='relu', name='x2')(x1)
    x3 = tf.keras.layers.Dense(units=16, activation='relu', name='x3')(x2)
    x4 = tf.keras.layers.Add(name='x4')([x1, x3])
    outputs = tf.keras.layers.Softmax(name='x5')(x4)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def non_sequential_simple_cut_start():
    inputs = tf.keras.Input(shape=[16], dtype=tf.dtypes.float32, name='x0')
    x1 = tf.keras.layers.Dense(units=16, activation='relu', name='x1')(inputs)
    x2 = tf.keras.layers.Dense(units=16, activation='relu', name='x2')(x1)
    x3 = tf.keras.layers.Add(name='x3')([inputs, x2])
    outputs = tf.keras.layers.Softmax(name='x4')(x3)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def non_sequential_simple_cut_start_and_end():
    inputs = tf.keras.Input(shape=[10], dtype=tf.dtypes.float32, name='x0')
    x1 = tf.keras.layers.Dense(units=16, activation='relu', name='x1')(inputs)
    x2 = tf.keras.layers.Dense(units=10, activation='relu', name='x2')(x1)
    outputs = tf.keras.layers.Add(name='x3')([inputs, x2])
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def non_sequential_medium():
    inputs = tf.keras.Input(shape=[10], dtype=tf.dtypes.float32, name='x0')
    x1 = tf.keras.layers.Dense(units=16, activation='relu', name='x1')(inputs)

    x20 = tf.keras.layers.Dense(units=16, activation='relu', name='x20')(x1)
    x30 = tf.keras.layers.Dense(units=16, activation='relu', name='x30')(x20)
    x21 = tf.keras.layers.Dense(units=16, activation='relu', name='x21')(x1)
    x31 = tf.keras.layers.Dense(units=16, activation='relu', name='x31')(x21)

    x4 = tf.keras.layers.Add(name='x4')([x30, x1, x31])

    x50 = tf.keras.layers.Dense(units=16, activation='relu', name='x50')(x4)
    x60 = tf.keras.layers.Dense(units=16, activation='relu', name='x60')(x50)
    x51 = tf.keras.layers.Dense(units=16, activation='relu', name='x51')(x4)
    x61 = tf.keras.layers.Dense(units=16, activation='relu', name='x61')(x51)

    x7 = tf.keras.layers.Add(name='x7')([x60, x61])
    outputs = tf.keras.layers.Softmax(name='x8')(x7)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def non_sequential_nested():
    inputs = tf.keras.Input(shape=[10], dtype=tf.dtypes.float32, name='x0')

    x10 = tf.keras.layers.Dense(units=16, activation='relu', name='x10')(inputs)
    x30 = tf.keras.layers.Dense(units=16, activation='relu', name='x30')(x10)

    x11 = tf.keras.layers.Dense(units=16, activation='relu', name='x11')(inputs)
    x20 = tf.keras.layers.Dense(units=16, activation='relu', name='x20')(x11)
    x21 = tf.keras.layers.Dense(units=16, activation='relu', name='x21')(x11)
    x31 = tf.keras.layers.Add(name='x31')([x20, x21])

    x4 = tf.keras.layers.Add(name='x4')([x30, x31])
    outputs = tf.keras.layers.Softmax(name='x5')(x4)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def non_sequential_nested_cut_branch_start():
    inputs = tf.keras.Input(shape=[10], dtype=tf.dtypes.float32, name='x0')

    x10 = tf.keras.layers.Dense(units=16, activation='relu', name='x10')(inputs)
    x20 = tf.keras.layers.Dense(units=16, activation='relu', name='x20')(x10)

    x11 = tf.keras.layers.Dense(units=16, activation='relu', name='x11')(inputs)
    x12 = tf.keras.layers.Dense(units=16, activation='relu', name='x12')(inputs)
    x21 = tf.keras.layers.Add(name='x21')([x11, x12])

    x3 = tf.keras.layers.Add(name='x3')([x20, x21])
    outputs = tf.keras.layers.Softmax(name='x4')(x3)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def non_sequential_complex():
    inputs = tf.keras.Input(shape=[16], dtype=tf.dtypes.float32, name='x0')

    x1 = tf.keras.layers.Dense(units=16, activation='relu', name='x1')(inputs)
    x21 = tf.keras.layers.Dense(units=16, activation='relu', name='x21')(x1)

    x20 = tf.keras.layers.Dense(units=16, activation='relu', name='x20')(inputs)
    x3 = tf.keras.layers.Add(name='x3')([inputs, x21])
    x41 = tf.keras.layers.Add(name='x41')([x20, x3])
    x51 = tf.keras.layers.Dense(units=16, activation='relu', name='x51')(x41)

    x42 = tf.keras.layers.Dense(units=16, activation='relu', name='x42')(x21)
    x52 = tf.keras.layers.Dense(units=16, activation='relu', name='x52')(x42)
    x6 = tf.keras.layers.Add(name='x6')([x51, x52])
    outputs = tf.keras.layers.Softmax(name='x7')(x6)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

