#!/usr/bin/env python

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.readlines()

with open('requirements_dev.txt') as requirements_file:
    test_requirements = [
        req for req in requirements_file.readlines()
        if 'git+' not in req
    ] + requirements

with open('requirements_addon.txt') as requirements_file:
    addon_requirements = requirements_file.readlines()

with open('requirements_viz.txt') as requirements_file:
    viz_requirements = requirements_file.readlines()

setup(
    name='Skills ML',
    version='2.1.0',
    description='Algorithms for Jobs/skills taxonomy creation',
    author="Center for Data Science and Public Policy",
    author_email='datascifellows@gmail.com',
    url='https://github.com/workforce-data-initiative/skills-ml',
    packages=find_packages(include=['skills_ml*']),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'tensorflow': addon_requirements,
        'viz': viz_requirements,
    },
    license="MIT license",
    keywords='nlp jobs skills onet',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
