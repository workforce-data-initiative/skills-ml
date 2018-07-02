from skills_ml.ontologies import Competency, Occupation, CompetencyOccupationEdge, CompetencyOntology
from unittest import TestCase
import json


class CompetencyTest(TestCase):
    def test_create(self):
        # Should be able to create a competency with the necessary inputs
        competency = Competency(identifier='123', name='communication', category='social skills')
        assert competency.identifier == '123'
        assert competency.name == 'communication'
        assert competency.category == 'social skills'

    def test_equivalence(self):
        # Two competencies with the same id should be equivalent
        competency = Competency(identifier='123', name='communication', category='social skills')
        competency_two = Competency(identifier='123', name='communication', category='social skills')
        assert competency == competency_two

    def test_add_parent(self):
        # Adding a parent should mutate the parents on self and the children on the parent
        communication = Competency(identifier='123', name='communication', category='social skills')
        listening = Competency(identifier='456', name='listening', category='social skills')
        communication.add_parent(listening)
        assert listening in communication.parents
        assert communication in listening.children

    def test_add_child(self):
        # Adding a child should mutate the children on self and the parents on the child
        communication = Competency(identifier='123', name='communication', category='social skills')
        listening = Competency(identifier='456', name='listening', category='social skills')
        listening.add_child(communication)
        assert listening in communication.parents
        assert communication in listening.children

    def test_jsonld(self):
        # The jsonld version of a competency should include all parent/child links, using the jsonld id
        communication = Competency(identifier='123', name='communication', category='social skills')
        listening = Competency(identifier='456', name='listening', category='social skills')
        listening.add_child(communication)
        assert listening.jsonld_full == {
            '@type': 'Competency',
            '@id': '456',
            'name': 'listening',
            'competencyCategory': 'social skills',
            'hasChild': [{'@type': 'Competency', '@id': '123'}],
            'isChildOf': [],
        }

    def test_from_jsonld(self):
        jsonld_input = {
            '@type': 'Competency',
            '@id': '456',
            'name': 'listening',
            'competencyCategory': 'social skills',
            'hasChild': [{'@type': 'Competency', '@id': '123'}],
            'isChildOf': [],
            'extra_kwarg': 'extra_value'
        }

        target_competency = Competency(identifier='456', name='listening', category='social skills', extra_kwarg='extra_value')
        target_competency.add_child(Competency(identifier='123'))
        competency = Competency.from_jsonld(jsonld_input)
        assert competency == target_competency


class OccupationTest(TestCase):
    def test_create(self):
        # Should be able to create an occupation with the necessary inputs
        occupation = Occupation(identifier='456', name='Civil Engineer')
        assert occupation.identifier == '456'
        assert occupation.name == 'Civil Engineer'

    def test_equivalence(self):
        # Two occupations with the same id should be equivalent
        occupation = Occupation(identifier='456', name='Civil Engineer')
        occupation_two = Occupation(identifier='456', name='Civil Engineer')
        assert occupation == occupation_two

    def test_add_parent(self):
        # Adding a parent should mutate the parents on self and the children on the parent
        civil_engineer = Occupation(identifier='12-3', name='Civil Engineer')
        engineers = Occupation(identifier='12', name='major group 12')
        civil_engineer.add_parent(engineers)
        assert engineers in civil_engineer.parents
        assert civil_engineer in engineers.children

    def test_add_child(self):
        # Adding a child should mutate the children on self and the parents on the child
        civil_engineer = Occupation(identifier='12-3', name='Civil Engineer')
        engineers = Occupation(identifier='12', name='major group 12')
        engineers.add_child(civil_engineer)
        assert engineers in civil_engineer.parents
        assert civil_engineer in engineers.children

    def test_jsonld(self):
        # The jsonld version of an occupation should include all parent/child links, using the jsonld id
        civil_engineer = Occupation(identifier='12-3', name='Civil Engineer')
        engineers = Occupation(identifier='12', name='major group 12')
        engineers.add_child(civil_engineer)
        assert civil_engineer.jsonld_full == {
            '@type': 'Occupation',
            '@id': '12-3',
            'name': 'Civil Engineer',
            'hasChild': [],
            'isChildOf': [{'@type': 'Occupation', '@id': '12'}],
        }

    def test_from_jsonld(self):
        jsonld_input = {
            '@type': 'Occupation',
            '@id': '12-3',
            'name': 'Civil Engineer',
            'hasChild': [],
            'isChildOf': [{'@type': 'Occupation', '@id': '12'}],
        }

        target_occupation = Occupation(identifier='12-3', name='Civil Engineer')
        target_occupation.add_parent(Occupation(identifier='12', name='major group 12'))
        occupation = Occupation.from_jsonld(jsonld_input)
        assert occupation == target_occupation


class CompetencyOccupationEdgeTest(TestCase):
    def test_create(self):
        # Should be able to create an edge with a competency and occupation
        competency = Competency(identifier='123', name='communication', category='social skills')
        occupation = Occupation(identifier='456', name='Civil Engineer')
        edge = CompetencyOccupationEdge(competency=competency, occupation=occupation)
        assert edge.competency == competency
        assert edge.occupation == occupation


    def test_jsonld(self):
        competency = Competency(identifier='111', name='communication', category='social skills')
        occupation = Occupation(identifier='123', name='Civil Engineer')
        edge = CompetencyOccupationEdge(competency=competency, occupation=occupation)
        assert edge.jsonld_full == {
            '@type': 'CompetencyOccupationEdge',
            '@id': 'competency=111&occupation=123',
            'competency': {
                '@type': 'Competency',
                '@id': '111'
            },
            'occupation': {
                '@type': 'Occupation',
                '@id': '123'
            }
        }

    def test_fromjsonld(self):
        jsonld_input = {
            '@type': 'CompetencyOccupationEdge',
            '@id': 'competency=111&occupation=123',
            'competency': {
                '@type': 'Competency',
                '@id': '111'
            },
            'occupation': {
                '@type': 'Occupation',
                '@id': '123'
            }
        }

        competency = Competency(identifier='111')
        occupation = Occupation(identifier='123')
        target_edge = CompetencyOccupationEdge(competency=competency, occupation=occupation)

        edge = CompetencyOccupationEdge.from_jsonld(jsonld_input)
        assert edge == target_edge

    
class OntologyTest(TestCase):
    def test_add_competency(self):
        # Should be able to add a competency to an ontology
        competency = Competency(identifier='123', name='communication', category='social skills')
        ontology = CompetencyOntology()
        ontology.add_competency(competency)
        assert len(ontology.competencies) == 1
        assert competency in ontology.competencies

    def test_add_occupation(self):
        # Should be able to add an occupation to an ontology
        occupation = Occupation(identifier='456', name='Civil Engineer')
        ontology = CompetencyOntology()
        ontology.add_occupation(occupation)
        assert len(ontology.occupations) == 1
        assert occupation in ontology.occupations

    def test_add_edge(self):
        # Should be able to add an edge between an occupation and a competency to an ontology
        occupation = Occupation(identifier='456', name='Civil Engineer')
        competency = Competency(identifier='123', name='communication', category='social skills')
        ontology = CompetencyOntology()
        ontology.add_edge(competency=competency, occupation=occupation)
        assert competency in ontology.competencies
        assert occupation in ontology.occupations
        assert len([edge for edge in ontology.edges if edge.occupation == occupation and edge.competency == competency]) == 1

    def test_filter_by(self):
        # Should be able to take an ontology and filter it by the edges, returning a new sub-ontology
        ontology = CompetencyOntology()
        comm = Competency(identifier='123', name='communication', category='social skills')
        python = Competency(identifier='999', name='python', category='Technologies')
        math = Competency(identifier='111', name='mathematics', category='Knowledge')
        science = Competency(identifier='222', name='science', category='Knowledge')

        civil_engineer = Occupation(identifier='123', name='Civil Engineer')
        ontology.add_competency(comm)
        ontology.add_competency(python)
        ontology.add_competency(math)
        ontology.add_competency(science)
        ontology.add_occupation(civil_engineer)
        ontology.add_edge(occupation=civil_engineer, competency=math)
        ontology.add_edge(occupation=civil_engineer, competency=science)

        tech_ontology = ontology.filter_by(lambda edge: edge.competency.category == 'Technologies')
        assert tech_ontology.competencies == {python}
        assert len(tech_ontology.occupations) == 0

        civil_engineer_ontology = ontology.filter_by(lambda edge: edge.occupation == civil_engineer)
        assert civil_engineer_ontology.competencies == {math, science}
        assert civil_engineer_ontology.occupations == {civil_engineer}

    def ontology(self):
        ontology = CompetencyOntology()
        comm = Competency(identifier='123', name='communication', category='social skills')
        python = Competency(identifier='999', name='python', category='Technologies')
        math = Competency(identifier='111', name='mathematics', category='Knowledge')
        science = Competency(identifier='222', name='science', category='Knowledge')

        civil_engineer = Occupation(identifier='123', name='Civil Engineer')
        ontology.add_competency(comm)
        ontology.add_competency(python)
        ontology.add_competency(math)
        ontology.add_competency(science)
        ontology.add_occupation(civil_engineer)
        ontology.add_edge(occupation=civil_engineer, competency=math)
        ontology.add_edge(occupation=civil_engineer, competency=science)
        return ontology

    def jsonld(self):
        return json.dumps({
            'occupations': [{
                '@type': 'Occupation',
                '@id': '123',
                'name': 'Civil Engineer',
                'hasChild': [],
                'isChildOf': [],
            }],
            'competencies': [
                {
                    '@type': 'Competency',
                    '@id': '111',
                    'name': 'mathematics',
                    'competencyCategory': 'Knowledge',
                    'hasChild': [],
                    'isChildOf': [],
                },
                {
                    '@type': 'Competency',
                    '@id': '123',
                    'name': 'communication',
                    'competencyCategory': 'social skills',
                    'hasChild': [],
                    'isChildOf': [],
                },
                {
                    '@type': 'Competency',
                    '@id': '222',
                    'name': 'science',
                    'competencyCategory': 'Knowledge',
                    'hasChild': [],
                    'isChildOf': [],
                },
                {
                    '@type': 'Competency',
                    '@id': '999',
                    'name': 'python',
                    'competencyCategory': 'Technologies',
                    'hasChild': [],
                    'isChildOf': [],
                },
            ],
            'edges': [
                {
                    '@type': 'CompetencyOccupationEdge',
                    '@id': 'competency=111&occupation=123',
                    'competency': {
                        '@type': 'Competency',
                        '@id': '111'
                    },
                    'occupation': {
                        '@type': 'Occupation',
                        '@id': '123'
                    }
                },
                {
                    '@type': 'CompetencyOccupationEdge',
                    '@id': 'competency=222&occupation=123',
                    'competency': {
                        '@type': 'Competency',
                        '@id': '222'
                    },
                    'occupation': {
                        '@type': 'Occupation',
                        '@id': '123'
                    }
                }
            ]
        }, sort_keys=True)

    def test_jsonld(self):
        assert self.ontology().jsonld == self.jsonld()

    def test_import_from_jsonld(self):
        assert CompetencyOntology.from_jsonld(self.jsonld()) == self.ontology()
