# respiration_extraction
## General 
1. This repo contains the code written for my bachelor thesis at the FAU in collaboration with the Machine Learning and Data Analytics Lab (MaD Lab).
2. Its structured in Datasets, Algorithms and Pipelines according to the tpcp-Concept from Küderle et al. (https://tpcp.readthedocs.io/en/latest/README.html).
3. It implements various Algorithms for extracting Respiratory Rate (RR) from other biomedical signals e.g. IMU and ECG. The algorithms were inspired from other papers, which are mentioned in the Documentation and/or the corresponding class names.



## Project structure

```
respiration_extraction
│   README.md
├── respiration_extraction  # The core library folder. All project-wide helper and algorithms go here
|
├── experiments  # The main folder for all experiements. Each experiment has its own subfolder
|   ├── experiment_1  # A single experiment (can be created with `poe experiment experiment_name`)
|   |   ├── notebooks  # All narrative notebooks belonging to the experiment.
|   |   ├── scripts  # Python scripts for this experiment
|   |   ├── helper  # A Python module with experiment specific helper functions
|   |
|   ├── experiment_2
|       ├── ...
|
├── tests  # Unit tests for the `respiration_extraction` library
|
├── data  # The main data folder. This is ignored in the `.gitignore` by default.
|
|   pyproject.toml  # The required python dependencies for the project
|   poetry.lock  # The frozen python dependencies to reproduce exact results
|
```

## Usage

This project was created using the mad-cookiecutter ds-base template.

To work with the project you need to install:

- [poetry](https://python-poetry.org/docs/#installation)
- [poethepoet](https://github.com/nat-n/poethepoet) in your global python env (`pip install poethepoet`)

Afterwards run:

```
poetry install
```

Then you can create a new experiment using:

```
poe experiment experiment_name
```


### Dependency management

All dependencies are manged using `poetry`.
Poetry will automatically create a new venv for the project, when you run `poetry install`.
Check out the [documentation](https://python- poetry.org/docs/basic-usage/) on how to add and remove dependencies.


### Jupyter Notebooks

To use jupyter notebooks with the project you need to add a jupyter kernel pointing to the venv of the project.
This can be done by running:

```
poe conf_jupyter
```

Afterwards a new kernel called `respiration_extraction` should be available in the jupyter lab / jupyter notebook interface.
Use that kernel for all notebooks related to this project.


You should also enable nbstripout, so that only clean versions of your notebooks get committed to git

```
poe conf_nbstripout
```


All jupyter notebooks should go into the `notebooks` subfolder of the respective experiment.
To make best use of the folder structure, the parent folder of each notebook should be added to the import path.
This can be done by adding the following lines to your first notebook cell:

```python
# Optional: Auto reloads the helper and the main respiration_extraction module
%load_ext autoreload
%autoreload 2

from respiration_extraction import conf_rel_path
conf_rel_path()
```

This allows to then import the helper and the script module belonging to a specific experiment as follows:

```
import helper
# or
from helper import ...
```

### Format and Linting

To ensure consistent code structure this project uses prospector, black, and isort to automatically check the code format.

```
poe format  # runs black and isort
poe lint # runs prospector
```

If you want to check if all code follows the code guidelines, run `poe check`.
This can be useful in the CI context


### Tests

All tests are located in the `tests` folder and can be executed by using `poe test`.
