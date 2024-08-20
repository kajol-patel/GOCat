# GOCat - Gene Ontology Categorizer

### Overview

**GOCat** is a tool designed to classify biological literature text into specific Gene Ontology (GO) categories. It supports two types of classifications:

* **Namespace Classification**: A multiclass classification that assigns the text to one of the GO namespaces.
* **"Is A" Classification**: A multilabel classification that categorizes the text based on "is a" relation categories within the GO hierarchy.

This tool is particularly useful for researchers and bioinformaticians involved in gene ontology annotations.

### Project Structure
```
.
├── Readme.md
├── dataset
│   └── go-basic.obo
├── experiments
│   ├── KNN1.png
│   ├── SVM1.png
│   ├── gocat_multiclass.ipynb
│   ├── gocat_multilabel_1.ipynb
│   ├── gocat_multilabel_2.ipynb
│   └── piechart.png
├── gocat_tool
│   ├── __pycache__
│   │   └── namespace_classifier.cpython-311.pyc
│   ├── is_a_classifier
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── is_a_classifier.cpython-311.pyc
│   │   ├── is_a_classifier.py
│   │   └── models
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       ├── random_forest.py
│   │       └── svm.py
│   └── namespace_classifier
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-311.pyc
│       │   ├── classifier.cpython-311.pyc
│       │   └── namespace_classifier.cpython-311.pyc
│       ├── models
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   ├── namespace_knn.py
│       │   ├── namespace_random_forest.py
│       │   └── namespace_svm.py
│       └── namespace_classifier.py
├── images
│   ├── KNN1.png
│   ├── SVM1.png
│   └── example.png
├── requirements.txt
├── run.py
└── testing
    ├── testing.py
    └── testing_isa.py
```

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
