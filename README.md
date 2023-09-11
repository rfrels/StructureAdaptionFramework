StructureAdaptionFramework: a framework for handling neuron-level and layer-level structure adaptions in
neural networks.

Copyright (C) 2023  Roman Frels, roman.frels@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Structure Adaption framework

If you are looking for a framework that automates neuron-level, layer-level and cell-level growing and pruning
for your research, look no further!

The structure adaption framework can:
- adapt the model architecture during training
- get all adaptable network structures in a convenient form
- remove neurons from layers and add neurons to layers (while specifying the newly added weights)
- remove or add (multiple) layers in complex arrangements either in sequence or parallel. 
- manage optimizer slots for you, while growing and pruning

For further details take a look at the documentation, as well as the provided examples.

For other licensing options please do approach me!

## Setup
Clone this repository and set up a virtual environment with python3.9. 
Then run the setup shell to install it. It will run the provided tests automatically. 

```bash
git clone 
cd StructureAdaptionFramework
bash run.sh
```

