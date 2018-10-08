from skills_ml.ontologies import Competency, Occupation, CompetencyOccupationEdge, CompetencyOntology, CompetencyFramework, research_hub_url
from skills_ml.storage import FSStore
from unittest import TestCase
import tempfile
import httpretty
import json


class CompetencyTest(TestCase):
    def test_create(self):
        # Should be able to create a competency with the necessary inputs
        competency = Competency(identifier='123', name='communication', categories=['social skills'])
        assert competency.identifier == '123'
        assert competency.name == 'communication'
        assert competency.categories[0] == 'social skills'

    def test_equivalence(self):
        # Two competencies with the same id should be equivalent
        competency = Competency(identifier='123', name='communication', categories=['social skills'])
        competency_two = Competency(identifier='123', name='communication', categories=['social skills'])
        assert competency == competency_two

    def test_add_parent(self):
        # Adding a parent should mutate the parents on self and the children on the parent
        communication = Competency(identifier='123', name='communication', categories=['social skills'])
        listening = Competency(identifier='456', name='listening', categories=['social skills'])
        communication.add_parent(listening)
        assert listening in communication.parents
        assert communication in listening.children

    def test_add_child(self):
        # Adding a child should mutate the children on self and the parents on the child
        communication = Competency(identifier='123', name='communication', categories=['social skills'])
        listening = Competency(identifier='456', name='listening', categories=['social skills'])
        listening.add_child(communication)
        assert listening in communication.parents
        assert communication in listening.children

    def test_jsonld(self):
        # The jsonld version of a competency should include all parent/child links, using the jsonld id
        communication = Competency(identifier='123', name='communication', categories=['social skills'])
        listening = Competency(identifier='456', name='listening', categories=['social skills'])
        listening.add_child(communication)
        assert listening.jsonld_full == {
            '@type': 'Competency',
            '@id': '456',
            'name': 'listening',
            'competencyCategory': ['social skills'],
            'hasChild': [{'@type': 'Competency', '@id': '123'}],
            'isChildOf': [],
        }

    def test_from_jsonld(self):
        jsonld_input = {
            '@type': 'Competency',
            '@id': '456',
            'name': 'listening',
            'competencyCategory': ['social skills'],
            'hasChild': [{'@type': 'Competency', '@id': '123'}],
            'isChildOf': [],
            'extra_kwarg': 'extra_value'
        }

        target_competency = Competency(identifier='456', name='listening', categories=['social skills'], extra_kwarg='extra_value')
        target_competency.add_child(Competency(identifier='123'))
        competency = Competency.from_jsonld(jsonld_input)
        assert competency == target_competency


class CompetencyFrameworkTest(TestCase):
    def test_create(self):
        competency = Competency(identifier='1', name='Doing Things', competencyText='The ability to do things')
        framework = CompetencyFramework(
            name='fundamentals_of_nothing',
            description='Fundamentals of Nothing',
            competencies=[competency]
        )
        assert len(framework) == 1
        assert framework['1'].name == 'Doing Things'

    def test_add(self):
        framework = CompetencyFramework(
            name='fundamentals_of_nothing',
            description='Fundamentals of Nothing',
        )
        competency = Competency(identifier='1', name='Doing Things', competencyText='The ability to do things')
        framework.add(competency)
        assert len(framework) == 1


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
        competency = Competency(identifier='123', name='communication', categories=['social skills'])
        occupation = Occupation(identifier='456', name='Civil Engineer')
        edge = CompetencyOccupationEdge(competency=competency, occupation=occupation)
        assert edge.competency == competency
        assert edge.occupation == occupation


    def test_jsonld(self):
        competency = Competency(identifier='111', name='communication', categories=['social skills'])
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
        competency = Competency(identifier='123', name='communication', categories=['social skills'])
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

    def test_add_occupation_merge(self):
        # Should be able to add an occupation that already exists, and it will merge the attributes
        first_child = Occupation(identifier='456', name='Civil Engineer')
        parent_occupation = Occupation(identifier='45', name='Engineers')
        ontology = CompetencyOntology()
        first_child.add_parent(parent_occupation)
        ontology.add_occupation(first_child)
        ontology.add_occupation(parent_occupation)

        parent_occupation = Occupation(identifier='45', name='Engineers')
        second_child = Occupation(identifier='457', name='Structural Engineer')
        second_child.add_parent(parent_occupation)
        ontology.add_occupation(second_child)
        ontology.add_occupation(parent_occupation)

        assert len(ontology.occupations) == 3
        assert len(list(ontology.filter_by(lambda edge: edge.occupation.identifier == '45').occupations)[0].children) == 2

    def test_add_competency_merge(self):
        # Should be able to add an competency that already exists, and it will merge the attributes
        # Should be able to add a competency to an ontology
        first_child = Competency(identifier='123', name='writing blog posts')
        parent_competency = Competency(identifier='12', name='communication')
        first_child.add_parent(parent_competency)
        ontology = CompetencyOntology()
        ontology.add_competency(first_child)
        ontology.add_competency(parent_competency)

        parent_competency = Competency(identifier='12', name='communication')
        second_child = Competency(identifier='124', name='public speaking')
        second_child.add_parent(parent_competency)
        ontology.add_competency(second_child)
        ontology.add_competency(parent_competency)

        assert len(ontology.competencies) == 3
        assert len(list(ontology.filter_by(lambda edge: edge.competency.identifier == '12').competencies)[0].children) == 2

    def test_add_edge(self):
        # Should be able to add an edge between an occupation and a competency to an ontology
        occupation = Occupation(identifier='456', name='Civil Engineer')
        competency = Competency(identifier='123', name='communication', categories=['social skills'])
        ontology = CompetencyOntology()
        ontology.add_edge(competency=competency, occupation=occupation)
        assert competency in ontology.competencies
        assert occupation in ontology.occupations
        assert len([edge for edge in ontology.edges if edge.occupation == occupation and edge.competency == competency]) == 1

    def test_filter_by(self):
        # Should be able to take an ontology and filter it by the edges, returning a new sub-ontology
        ontology = CompetencyOntology()
        comm = Competency(identifier='123', name='communication', categories=['social skills'])
        python = Competency(identifier='999', name='python', categories=['Technologies'])
        math = Competency(identifier='111', name='mathematics', categories=['Knowledge'])
        science = Competency(identifier='222', name='science', categories=['Knowledge'])

        civil_engineer = Occupation(identifier='123', name='Civil Engineer')
        ontology.add_competency(comm)
        ontology.add_competency(python)
        ontology.add_competency(math)
        ontology.add_competency(science)
        ontology.add_occupation(civil_engineer)
        ontology.add_edge(occupation=civil_engineer, competency=math)
        ontology.add_edge(occupation=civil_engineer, competency=science)

        tech_ontology = ontology.filter_by(lambda edge: 'Technologies' in edge.competency.categories)
        assert tech_ontology.competencies == {python}
        assert len(tech_ontology.occupations) == 0

        civil_engineer_ontology = ontology.filter_by(lambda edge: edge.occupation == civil_engineer)
        assert civil_engineer_ontology.competencies == {math, science}
        assert civil_engineer_ontology.occupations == {civil_engineer}

    def ontology(self):
        ontology = CompetencyOntology(name='Test Ontology')
        comm = Competency(identifier='123', name='communication', categories=['social skills'])
        python = Competency(identifier='999', name='python', categories=['Technologies'])
        math = Competency(identifier='111', name='mathematics', categories=['Knowledge'])
        science = Competency(identifier='222', name='science', categories=['Knowledge'])

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
            'name': 'Test Ontology',
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
                    'competencyCategory': ['Knowledge'],
                    'hasChild': [],
                    'isChildOf': [],
                },
                {
                    '@type': 'Competency',
                    '@id': '123',
                    'name': 'communication',
                    'competencyCategory': ['social skills'],
                    'hasChild': [],
                    'isChildOf': [],
                },
                {
                    '@type': 'Competency',
                    '@id': '222',
                    'name': 'science',
                    'competencyCategory': ['Knowledge'],
                    'hasChild': [],
                    'isChildOf': [],
                },
                {
                    '@type': 'Competency',
                    '@id': '999',
                    'name': 'python',
                    'competencyCategory': ['Technologies'],
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
        assert CompetencyOntology(jsonld_string=self.jsonld()) == self.ontology()

    @httpretty.activate
    def test_import_from_url(self):
        url = 'http://testurl.com/ontology.json'
        httpretty.register_uri(
            httpretty.GET,
            url,
            body=self.jsonld(),
            content_type='application/json'
        )
        assert CompetencyOntology(url=url) == self.ontology()

    @httpretty.activate
    def test_import_from_researchhub(self):
        url = research_hub_url('testontology')
        httpretty.register_uri(
            httpretty.GET,
            url,
            body=self.jsonld(),
            content_type='application/json'
        )
        assert CompetencyOntology(research_hub_name='testontology') == self.ontology()

    def test_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FSStore(temp_dir)
            self.ontology().save(storage)
            assert CompetencyOntology(jsonld_string=storage.load('Test Ontology.json')) == self.ontology()

    def test_competency_counts_per_occupation(self):
        assert sorted(self.ontology().competency_counts_per_occupation) == [2]

    def test_occupation_counts_per_competency(self):
        assert sorted(self.ontology().occupation_counts_per_competency) == [0, 0, 1, 1]

    def test_print_summary(self):
        # literally just want to make sure that this function doesn't error out
        # due to bad interpolation or something. no asserts
        self.ontology().print_summary_stats()
