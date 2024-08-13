# GOCat - Gene Ontology Categorizer

### Overview

**GOCat** is a tool designed to classify biological literature text into specific Gene Ontology (GO) categories. It supports two types of classifications:

* **Namespace Classification**: A multiclass classification that assigns the text to one of the GO namespaces.
* **"Is A" Classification**: A multilabel classification that categorizes the text based on "is a" relation categories within the GO hierarchy.

This tool is particularly useful for researchers and bioinformaticians involved in gene ontology annotations.

### Project Structure

.
├── [README.md](README.md)               # Project overview and usage instructions
├── [requirements.txt](requirements.txt) # Dependencies for the project
├── [setup.py](setup.py)                 # Setup script
├── [my_project/](my_project)            # Source code directory
│   ├── [__init__.py](my_project/__init__.py)         # Package initialization
│   ├── [module1.py](my_project/module1.py)          # Core functionalities
│   ├── [module2.py](my_project/module2.py)          # Additional features
│   └── [utils/](my_project/utils)                   # Helper utilities
│       └── [helper.py](my_project/utils/helper.py)  # Helper functions
└── [tests/](tests)                     # Unit tests
    ├── [test_module1.py](tests/test_module1.py)     # Tests for module1
    └── [test_module2.py](tests/test_module2.py)     # Tests for module2


### Installation

1. Create a virtual environment with the command
`python -m venv .venv`
2. Activate the virtual environment
`source .venv/bin/activate`
3. Install Dependencies using 
`pip install -r requirements.txt`

### Usage

After installing the necessary dependencies, you can use the tool by running the run.py file using the below command.

`python3 run.py`

 The script will prompt you for various inputs via the command line and then provide the classification results.

**Example**

![Example on how to run run.py in CLI](images/example.png)
