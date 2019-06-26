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
* `flow_discretize.py` - implementation of the discretization of flags and bytes followed by combining into a single discrete feature 

### Botnet profiling

* `Profiling.ipynb` - contains the full analysis. note that it takes some time and ram
to run

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

## Data :floppy_disk:
If unable to clone the repository, download the [CTU-13 dataset](https://www.stratosphereips.org/datasets-ctu13/) as follows:

* for Task 1+2 [Scenario 6](https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47/) file `capture20110816.pcap.netflow.labeled`
* for the other tasks [Scenario 10](https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/) file `capture20110818.pcap.netflow.labeled`

> :exclamation: After downloading the files, place them into the `data\` folder

## Instructions for cloning :memo:
The data files were uploaded using [Git LFS](https://git-lfs.github.com/) being over 100MB. Git LFS is needed to clone the repository. Install it manually or try to use `downlopad_data_files.sh`.

## Installation :computer:
The scripts can be run in [Anaconda](https://www.anaconda.com/download/) Windows/Linux environment.

You need to create an Anaconda :snake: `python 3.6` environment named `cyber3`.
Inside that environment some addition packages needs to be installed. Run the following commands inside Anaconda Prompt ⌨:
```shell
(base) conda create -n cyber3 python=3.6 anaconda
(base) conda activate cyber3
(cyber3) conda install -c conda-forge tqdm 
(cyber3) conda install -c conda-forge mmh3
```
