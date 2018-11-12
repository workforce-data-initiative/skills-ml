from skills_ml.ontologies.clustering import Clustering
from skills_ml.ontologies.onet import Onet

import unittest
import collections
from descriptors import cachedproperty
import pytest

# onet = Onet()

@pytest.mark.usefixtures("onet")
def test_basic(onet):
    # Assuming we have a bunch of concepts and associated entities, we call
    # each concept and associated entities a cluster.

    # For example, an major group in O*Net is a concept and the associated entities
    # are occupations within that major group

    major_group_37_concept= list(onet.filter_by(lambda edge: edge.occupation.identifier == '37').occupations)[0]
    major_group_37_entities = list(onet.filter_by(lambda edge: edge.occupation.identifier[:3] == '37-').occupations)

    major_group_35_concept= list(onet.filter_by(lambda edge: edge.occupation.identifier == '35').occupations)[0]
    major_group_35_entities = list(onet.filter_by(lambda edge: edge.occupation.identifier[:3] == '35-').occupations)

    occupation_clustering = Clustering(
            name="test_major_group_occupations",
            key_transform_fn=lambda key: key.name
    )

    occupation_clustering[major_group_35_concept] = major_group_35_entities
    occupation_clustering[major_group_37_concept] = major_group_37_entities

    # Add cluster
    assert len(occupation_clustering) == 2
    assert set(occupation_clustering.keys()) == set([major_group_37_concept.name, major_group_35_concept.name])
    assert occupation_clustering["Building and Grounds Cleaning and Maintenance"] == major_group_37_entities
    assert occupation_clustering.map_raw_key["Building and Grounds Cleaning and Maintenance"] == major_group_37_concept
    assert occupation_clustering["Food Preparation and Serving Related Occupations"] == major_group_35_entities
    assert occupation_clustering.map_raw_key["Food Preparation and Serving Related Occupations"] == major_group_35_concept

    # Delete
    del occupation_clustering["Food Preparation and Serving Related Occupations"]
    assert len(occupation_clustering) == 1

    # Iterable
    assert isinstance(occupation_clustering, collections.Iterable)

