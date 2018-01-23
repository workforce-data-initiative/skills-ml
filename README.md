skill-ml
=========

[![Build Status](https://travis-ci.org/workforce-data-initiative/skills-ml.svg?branch=master)](https://travis-ci.org/workforce-data-initiative/skills-ml)
[![Code Coverage](https://codecov.io/gh/workforce-data-initiative/skills-ml/branch/master/graph/badge.svg)](https://codecov.io/gh/workforce-data-initiative/skills-ml)
[![Updates](https://pyup.io/repos/github/workforce-data-initiative/skills-ml/shield.svg)](https://pyup.io/repos/github/workforce-data-initiative/skills-ml/)
[![Python 3](https://pyup.io/repos/github/workforce-data-initiative/skills-ml/python-3-shield.svg)](https://pyup.io/repos/github/workforce-data-initiative/skills-ml/)
[![PyPI](https://img.shields.io/pypi/v/skills-ml.svg)]()


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


```bash
pip install skills-ml
```

### 3. Import skills_ml
```python
import skills_ml
```

skills-ml doesn't have a tutorial yet, but here are some useful places to start.
- There are a couple of examples of specific uses of components to perform specific tasks in [examples](examples/).
- Check out the descriptions of different algorithm types in [algorithms/](skills_ml/algorithms/) and look at any individual directories that match what you'd like to do (e.g. skill extraction, job title normalization)
- [skills-airflow](https://github.com/workforce-data-initiative/skills-airflow) is the open-source production system that uses skills-ml algorithms in an Airflow pipeline to generate open datasets

Structure
----------
- [algorithms/](skills_ml/algorithms/) - Core algorithmic module. Each submodule is meant to contain a different type of component, such as a job title normalizer or a skill tagger, with a common interface so different pipelines can try out different versions of the components.
- [datasets/](skills_ml/datasets/) - Wrappers for interfacing with different datasets, such as ONET, Urbanized Area.
- [evaluation/](skills_ml/evaluation/) - Code for testing different components against each other.


Contributors
----------
Kwame Porter Robinson - [Github](https://github.com/robinsonkwame)
Eddie Lin - [Github](https://github.com/tweddielin)
Tristan Crockett - [Github](https://github.com/thcrock)


License
-------
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
