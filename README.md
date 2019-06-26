# Cyber Botnet assignment 3 :japanese_goblin:

Code for Group 66 python implementation of Cyber Data Analytics assigment 3 CS4035. :lock:

Team members:

 * [Andrei Simion-Constantinescu](https://www.linkedin.com/in/andrei-simion-constantinescu/)
 * [Mihai Voicescu](https://github.com/mihai1voicescu)
 
## Project structure :open_file_folder:
The structure of the project is presented per task:


### Sampling

* `reservoir_sampling.py` - implementation for reservoir sampling with testing for multiple reservoir sizes

### Sketching

* `countminsketch.py` - contains the `CountMinSketch` class.
* `CountMinSketch.ipynb` - the actual analysis and plots.

### Flow data discretization

* `Flow visualization.ipynb` - notebook for visualizing different features for the infected host
* `flow_visualization_utils.py` - helper functions for generating the plots from notebook

### Botnet profiling

* `Profiling.ipynb` - contains the full analysis. note that it takes some time and ram
to run

### Flow classification
* `flow_classification.py` - train and test Random Forest classifier for identifying a netflow probability of being a botnet


### Bonus
Included in the `Profiling.ipynb` and `flow_classification.py`


## Downloading the data-set when using `git clone`
In case you somehow do not have the data-set please either use the `downlopad_data_files.sh`
or manually install git lfs.

## Requirements
We use `python3`. The requirements can be installed from requirements.txt
or using `conda`.


