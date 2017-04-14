skill-ml
=========

[![Build Status](https://travis-ci.org/workforce-data-initiative/skills-ml.svg?branch=master)](https://travis-ci.org/workforce-data-initiative/skills-ml)
[![Code Coverage](https://codecov.io/gh/workforce-data-initiative/skills-ml/branch/master/graph/badge.svg)](https://codecov.io/gh/workforce-data-initiative/skills-ml)

**Open Skills Project - Machine Learning**

This is the library for the methods usable by the Open Skills API, including processing algorithms and utilities for computing our jobs and skills taxonomy.



Quick Start
-----------
### 1. Virtualenv
**skills-ml** depends on python3, so create a virtual environment using a python3 executable.

```bash
virtualenv venv -p /usr/bin/python3
``` 
Activate your virtualenv

```bash    
source venv/bin/activate
```

### 2. Installation

There are two ways to install **skills-ml**. You can either install from the source code or install with `pip install`

#### Install from source code

**skills-ml** is available through cloning the repository. First you need to install all the requirements.

```bash
git clone https://github.com/workforce-data-initiative/skills-ml.git
cd skills-ml
pip install requirements.txt
pip install requirements_dev.txt
```

Then install with `setup.py`

```bash  
python setup.py install
```

#### Install with `pip`

Or use `pip install`
    
```bash
pip install git+git://github.com/workforce-data-initiative/skills-ml.git@master
```

### 3. Import skills_ml
```python
from sills_ml import sills_ml
```

Structure
----------
- [algorithms/](https://github.com/workforce-data-initiative/skills-ml/tree/documentation/skills_ml/algorithms) - Core algorithmic module. Each subdirectory is meant to contain a different type of component, such as a job title normalizer or a skill tagger, with a common interface so different pipelines can try out different versions of the components.
- [api_sync/](https://github.com/workforce-data-initiative/skills-ml/tree/documentation/skills_ml/api_sync) - Module to manage integrating workforce data into a relational database suitable for powering the Open SkillsAPI.
- [datasets/](https://github.com/workforce-data-initiative/skills-ml/tree/documentation/skills_ml/datasets) - Wrappers for interfacing with different datasets, such as ONET, Urbanized Area.
- [evaluation/](https://github.com/workforce-data-initiative/skills-ml/tree/documentation/skills_ml/evaluation) - Code for testing different components against each other.

License
-------
This project is licensed under the MIT License - see the `LICENSE.md` file for details.