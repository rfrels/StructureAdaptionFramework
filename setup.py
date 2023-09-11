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

"""Structure Adaption Framework setup."""

from setuptools import setup, find_packages

setup(
    name='StructureAdaptionFramework',
    version='0.1',
    author='Roman Frels',
    author_email='roman.frels@gmail.com',
    description='a framework for handling neuron-level and layer-level structure adaptions in neural networks',
    long_description='a framework for handling neuron-level and layer-level structure adaptions in neural networks',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_package_name',
    packages=find_packages(),
    install_requires=[
        'protobuf==3.20.1',
        'bidict',
        'ml_collections',
        'numpy',
        'tensorflow==2.7.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Researchers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU Affero General Public License',
        'Programming Language :: Python :: 3.9',
    ],
)


