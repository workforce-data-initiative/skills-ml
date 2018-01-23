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

skills-ml is a library of different standalone algorithms side-by-side, so as such there isn't a beginning tutorial. There are, however, a couple of examples of specific uses of components in [examples](examples/), which are helpful. Otherwise check out the [algorithms/](skills_ml/algorithms/) directory and check out any individual directories that match what you'd like to do (e.g. skill extraction, job title normalization), or [skills-airflow](https://github.com/workforce-data-initiative/skills-airflow) for an example of a production system using skills-ml.

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
