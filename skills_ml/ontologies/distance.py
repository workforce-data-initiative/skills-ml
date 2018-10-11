from skills_ml.ontologies.base import Occupation, CompetencyOntology
import numpy as np

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


