from skills_ml.ontologies.base import Occupation, CompetencyOntology
from collections import MutableMapping, KeysView
import numpy as np

class KeysViewOnlyKeys(KeysView):
    # Subclass that just changes the representation.
    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"


class Clustering(MutableMapping):
    """ A clustering object acting like a dictionary which key is a cluster concept
    and value is a list of entities associated to that cluster

    """
    def __init__(self,
            name,
            key_transform_fn=lambda concept: concept,
            value_transform_fn=lambda entity: entity,):
        self.name = name
        self.store = dict()
        self.transformed = dict()
        self.map_original_key = dict()
        self.key_transform_fn = key_transform_fn
        self.value_transform_fn = value_transform_fn

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value
        self.map_original_key[self.__keytransform__(key)] = key

    def __getitem__(self, key):
        return self.store[key]

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return self.key_transform_fn(key)

    def transform(self):
        for concept, entities in self.items():
            self.transformed[concept] = list(map(self.value_transform_fn, entities))

    def keys(self):
        return KeysViewOnlyKeys(self)


def jaccard_distance(u:set, v:set):
    try:
        return 1 - float(len(u & v)) / len(u | v)
    except ZeroDivisionError:
        if len(u) == 0 and len(v) == 0:
            return 1.0
        else:
            return 1.0


def occupation_distance(occ1: Occupation, occ2: Occupation, ontology:CompetencyOntology, average=True):
    competency_categories = ontology.competency_categories
    output = {}
    occ1 = occ1[0]
    occ2 = occ2[0]
    for c in competency_categories:
        c1 = ontology.filter_by(lambda edge: edge.occupation.identifier == occ1.identifier).competencies
        c2 = ontology.filter_by(lambda edge: edge.occupation.identifier == occ2.identifier).competencies
        output[c] = jaccard_distance(set(filter(lambda x: c in x.categories, c1)), set(filter(lambda x: c in x.categories, c2)))

    if average:
        return sum(output.values()) / len(competency_categories)
    else:
        return output


def jaccard_competency_distance(c1, c2, competency_categories, average=True):
    output = {}
    for c in competency_categories:
        output[c] = jaccard_distance(set(filter(lambda x: c in x.categories, c1)), set(filter(lambda x: c in x.categories, c2)))
    if average:
        return sum(output.values()) / len(competency_categories)
    else:
        return output


