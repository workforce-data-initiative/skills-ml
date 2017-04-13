skill-ml
=========

[![Build Status](https://travis-ci.org/workforce-data-initiative/skills-ml.svg?branch=master)](https://travis-ci.org/workforce-data-initiative/skills-ml)
[![Code Coverage](https://codecov.io/gh/workforce-data-initiative/skills-ml/branch/master/graph/badge.svg)](https://codecov.io/gh/workforce-data-initiative/skills-ml)

**Open Skills Project - Machine Learning**

This is the library for the methods usable by the Open Skills API, including processing algorithms and utilities for computing our jobs and skills taxonomy.



Quick Start
-----------
### 1. Clone repository
skills-ml is available through cloning the repository and then working from the repository root.

    git clone https://github.com/workforce-data-initiative/skills-ml.git 
    cd skills-ml 

### 2. Virtualenv
skills-ml depends on python3, so create a virtual environment using a python3 executable.

    virtualenv venv -p /usr/bin/python3
    
### 3. Python requirements
Activate your virtualenv and install requirements.

	source venv/bin/activate 
	pip install requirements.txt 
	pip install requirements_dev.txt

### 4. Configure s3 buckets and prefixes
Configuration of external data sources, such as s3, is controlled via a config.yaml file in the root directory of the project. Copy `example_config.yaml` and modify it to match the source buckets and paths for the data you wish to use (such as job listings, or ONET extracts).

Structure
----------
### 1. Algorithms

### 2. 