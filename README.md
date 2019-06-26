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

TODO Mihai

### Flow data discretization

* `Flow visualization.ipynb` - notebook for visualizing different features for the infected host
* `flow_visualization_utils.py` - helper functions for generating the plots from notebook
* `flow_discretize.py` - implementation of the discretization of flags and bytes followed by combining into a single discrete feature 

### Botnet profiling

TODO Mihai

### Flow classification
* `flow_classification.py` - train and test Random Forest classifier for identifying a netflow probability of being a botnet

### Bonus
* `bonus.py` - implementation of the generation method for adversarial data

> :exclamation: The actual testing using adversarial data is in the files corresponding to profiling and classification tasks

#### Others
* `logger.py` - logging system for generating folders initial structure and saving application logs to HTML files 
* `utils.py` - helper functions used for multiple tasks
* `config.txt` - configuration file

#### Additional folders
* `data\` - for storing data files with BATADAL datasets
* `output\` - for storing plots at high resolution (**Better to be inspected if the ones from the report are too small due to page limit**)
* `logs\` - for storing a couple of logs files referred in the report
