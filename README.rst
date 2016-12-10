===============================
skills-ml
===============================

.. image:: https://travis-ci.org/workforce-data-initiative/skills-ml.svg?branch=master
        :target: https://travis-ci.org/workforce-data-initiative/skills-ml

.. image:: https://codecov.io/gh/workforce-data-initiative/skills-ml/branch/master/graph/badge.svg
	 :target: https://codecov.io/gh/workforce-data-initiative/skills-ml
	 :alt: Code Coverage

Open Skills Project - Machine Learning

This is the home for the methods and pipelines responsible for creating a skills and jobs ontology, usable by the Open Skills API.

Quick Start
-----------------

1. Clone repository
~~~~~~~~~~~~~~~~~~~~~~
skills-ml is available through cloning the repository and then working from the repository root.

`git clone https://github.com/workforce-data-initiative/skills-ml.git && cd skills-ml`

2. Virtualenv
~~~~~~~~~~~~~~~~~~~~~~
skills-ml depends on python3, so create a virtual environment using a python3 executable.

`virtualenv venv -p /usr/bin/python3`

3. Airflow
~~~~~~~~~~~~~~~~~~~~~~
Although many of the algorithmic components can be used as pure Python classes, the main way of running this code is on Airflow. There are a number of things to get Airflow up and running, but the script below will launch Airflow with a reasonable default configuration for this repository, and even start the webserver.

`sh tests/basic_deploy_skills_ml.sh`

The script will prompt you to use the Airflow UI to input S3 credentials. That should be done at this stage.

4. Configure s3 buckets and prefixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration of external data sources, such as s3, is controlled via a config.yaml file in the root directory of the project. Copy example_config.yaml to example_config.yaml and modify it to match the source buckets and paths for the data you wish to use (such as job listings, or ONET extracts).

5. Run whatever DAGs or tasks you like
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For instance, this code will test a sample ONET title extractor:
`PYTHONPATH='.' airflow test simple_machine_learning title_extract 2016-01-01`
More documentation is forthcoming on the different DAGs available and the methods that they implement

Structure
-----------------

* algorithms/ - Core algorithmic components. Each subdirectory is meant to contain a different type of component, such as a job title normalizer or a skill tagger, with a common interface so different pipelines can try out different versions of the components.
* evaluation/ - Code for testing different components against each other
* datasets.py - Wrappers for interfacing with different datasets, such as ONET
* dags/ - Pipelines in Airflow DAG form
* tests/ - A test suite
* utils/ - Common utilities
