from typing import Callable, Text, List
from collections import MutableMapping, KeysView
import json
from statistics import median, mean
from functools import total_ordering
import itertools
import requests


def research_hub_url(name: Text) -> Text:
    """Retrieve the current Research Hub URL for ontologies"""
    return 'https://open-skills-datasets.s3-us-west-2.amazonaws.com/v3/ontologies/' + name + '.json'


@total_ordering
class Competency(object):
    """Represents a competency not necessarily tied to an ontology

    Args:
        identifier: A unique identifier for this competency. Choose the identifier wisely as it will be used for equivalence with other competency objects
        name: A name for the competency (e.g. Microsoft Office)
        categories: Optional text categories for the competency that is not a higher-level competency itself
    """

    def __init__(self, identifier: Text, name: Text=None, categories: List[Text]=None, **kwargs):
        self.identifier = identifier
        self.name = name or ''
        self.categories = categories or []
        self.other_attributes = kwargs
        self.children = set()
        self.parents = set()

    @classmethod
    def from_jsonld(cls, jsonld_input):
        extra_kwargs = dict((key, jsonld_input[key]) for key in jsonld_input.keys() if key not in {'@type', '@id', 'name', 'competencyCategory', 'hasChild', 'isChildOf'})
        obj = cls(
            identifier=jsonld_input['@id'],
            name=jsonld_input.get('name', ''),
            categories=jsonld_input.get('competencyCategory', None),
            **extra_kwargs
        )
        for jsonld_child_obj in jsonld_input.get('hasChild', []):
            obj.add_child(cls(identifier=jsonld_child_obj['@id']))
        for jsonld_parent_obj in jsonld_input.get('isChildOf', []):
            obj.add_parent(cls(identifier=jsonld_parent_obj['@id']))
        return obj

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return NotImplemented
        return self.identifier == other.identifier

    def __lt__(self, other):
        if not isinstance(self, other.__class__):
            return NotImplemented
        return self.identifier < other.identifier

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        return f'Competency(identifier={self.identifier}, name={self.name}, categories={self.categories}, {self.other_attributes})'

    @property
    def jsonld_id(self):
        return {
            '@type': 'Competency',
            '@id': self.identifier
        }

    @property
    def jsonld_full(self):
        attributes = {
            '@type': 'Competency',
            '@id': self.identifier,
            'name': self.name,
            'competencyCategory': self.categories,
            'hasChild': [child.jsonld_id for child in self.children],
            'isChildOf': [parent.jsonld_id for parent in self.parents],
        }
        for key, value in self.other_attributes.items():
            attributes[key] = value
        return attributes

    def add_child(self, child):
        if not isinstance(child, Competency):
            raise ValueError('All children of a Competency must be Competencies themselves')
        if child not in self.children:
            self.children.add(child)
            child.add_parent(self)

    def add_parent(self, parent):
        if not isinstance(parent, Competency):
            raise ValueError('All parents of a Competency must be Competencies themselves')
        if parent not in self.parents:
            self.parents.add(parent)
            parent.add_child(self)


@total_ordering
class Occupation(object):
    """Represents an occupation that may or may not be part of an ontology

    Args:
        identifier: A unique identifier for this occupation. Choose the identifier wisely as it will be used for equivalence with other occupation objects
        name: A name for the occupation (e.g. Civil Engineer)
    """

    def __init__(self, identifier, name=None, **kwargs):
        self.identifier = identifier
        self.name = name or ''
        self.other_attributes = kwargs
        self.children = set()
        self.parents = set()

    @classmethod
    def from_jsonld(cls, jsonld_input):
        extra_kwargs = dict((key, jsonld_input[key]) for key in jsonld_input.keys() if key not in {'@type', '@id', 'name', 'hasChild', 'isChildOf'})
        obj = cls(
            identifier=jsonld_input['@id'],
            name=jsonld_input.get('name', ''),
            **extra_kwargs
        )
        for jsonld_child_obj in jsonld_input.get('hasChild', []):
            obj.add_child(cls(identifier=jsonld_child_obj['@id']))
        for jsonld_parent_obj in jsonld_input.get('isChildOf', []):
            obj.add_parent(cls(identifier=jsonld_parent_obj['@id']))
        return obj

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return NotImplemented
        return self.identifier == other.identifier

    def __lt__(self, other):
        if not isinstance(self, other.__class__):
            return NotImplemented
        return self.identifier < other.identifier

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        return f'Occupation(identifier={self.identifier}, name={self.name}, {self.other_attributes})'

    @property
    def jsonld_id(self):
        return {
            '@type': 'Occupation',
            '@id': self.identifier
        }

    @property
    def jsonld_full(self):
        attributes = {
            '@type': 'Occupation',
            '@id': self.identifier,
            'name': self.name,
            'hasChild': [child.jsonld_id for child in self.children],
            'isChildOf': [parent.jsonld_id for parent in self.parents],
        }
        for key, value in self.other_attributes.items():
            attributes[key] = value
        return attributes


    def add_child(self, child):
        if not isinstance(child, Occupation):
            raise ValueError('All children of a Occupation must be Occupations themselves')
        if child not in self.children:
            self.children.add(child)
            child.add_parent(self)

    def add_parent(self, parent):
        if not isinstance(parent, Occupation):
            raise ValueError('All parents of a Occupation must be Occupations themselves')
        if parent not in self.parents:
            self.parents.add(parent)
            parent.add_child(self)


class DummyOccupation(Occupation):
    def __init__(self):
        super().__init__('0', '')

class DummyCompetency(Competency):
    def __init__(self):
        super().__init__('0', '')


class CompetencyOccupationEdge(object):
    def __init__(self, competency, occupation, identifier=None):
        self.competency = competency
        self.occupation = occupation
        self.identifier = f'competency={competency.identifier}&occupation={occupation.identifier}'
        if identifier:
            assert identifier == self.identifier

    def __repr__(self):
        return f'CompetencyOccupationEdge(competency={self.competency}, occupation={self.occupation})'

    @classmethod
    def from_jsonld(cls, jsonld_input):
        obj = cls(
            identifier=jsonld_input['@id'],
            competency=Competency.from_jsonld(jsonld_input['competency']),
            occupation=Occupation.from_jsonld(jsonld_input['occupation'])
        )
        return obj

    @property
    def jsonld_id(self):
        return {
            '@type': 'CompetencyOccupationEdge',
            '@id': self.identifier
        }

    @property
    def jsonld_full(self):
        return {
            '@type': 'CompetencyOccupationEdge',
            '@id': self.identifier,
            'competency': self.competency.jsonld_id,
            'occupation': self.occupation.jsonld_id
        }

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.competency == other.competency\
                and self.occupation == other.occupation

    def __hash__(self):
        return hash(self.competency) + hash(self.occupation)


class CompetencyFramework(MutableMapping):
    """A list of competencies and metadata about them

    Implements MutableMapping, so the competencies may be interacted with as a dictionary.
    """

    def __init__(self, name=None, description=None, competencies=None):
        self.name = name or ''
        self.description = description or ''
        self._competencies = dict()
        if competencies:
            self.competencies = competencies

    def __setitem__(self, key, value):
        self._competencies[key] = value

    def __getitem__(self, key):
        return self._competencies[key]

    def __delitem__(self, key):
        del self._competencies[key]

    def __iter__(self):
        return iter(self._competencies)

    def __len__(self):
        return len(self._competencies)

    def add(self, value):
        if value.identifier in self._competencies:
            raise ValueError(f"{value} already in framework")
        self._competencies[value.identifier] = value

    @property
    def competencies(self):
        return self._competencies

    @competencies.setter
    def competencies(self, competencies):
        for competency in competencies:
            self._competencies[competency.identifier] = competency


class CompetencyOntology(object):
    """An ontology of competencies and occupations (each referred to as nodes) and the edges between them

    Can be initialized with:
        - a JSON-LD string, in which case all nodes and edges will be initialized from the parsed JSON-LD
        - a URL, in which case the URL is presumed to contain JSON-LD, parsed, and all nodes and edges will be initialized from the parsed JSON-LD
        - a research hub name, in which case the Open Skills Research Hub is presumed to host an ontology by that name. a URL is built based on the name and the nodes and edges will be initialized from the parsed JSON-LD
        - a set of edges, in which case all nodes will be initialized with any that are present in the edge list
    """
    name = 'unnamed_ontology'

    def __init__(
        self,
        edges=None,
        name=None,
        competency_name=None,
        competency_description=None,
        research_hub_name=None,
        url=None,
        jsonld_string=None
    ):
        self.name = name
        self._initialize_empty(competency_name, competency_description)
        if jsonld_string:
            self._build_from_jsonld(jsonld_string)
        elif url:
            self._build_from_url(url)
        elif research_hub_name:
            self._build_from_research_hub(research_hub_name)
        elif edges:
            self._build_from_edges(edges)

    def _initialize_empty(self, competency_framework_name, competency_framework_description):
        self.competency_framework = CompetencyFramework(
            name=competency_framework_name,
            description=competency_framework_description,
        )
        self._occupations = dict()
        self._competency_occupation_edges = set()

    def _build_from_research_hub(self, name):
        self._build_from_url(research_hub_url(name))

    def _build_from_url(self, url):
        jsonld_string = requests.get(url).content
        self._build_from_jsonld(jsonld_string)

    def _build_from_jsonld(self, jsonld_string: Text):
        jsonld_input = json.loads(jsonld_string)
        self.name = jsonld_input.get('name', 'unnamed ontology')
        for competency_jsonld in jsonld_input['competencies']:
            self.add_competency(Competency.from_jsonld(competency_jsonld))
        for occupation_jsonld in jsonld_input['occupations']:
            self.add_occupation(Occupation.from_jsonld(occupation_jsonld))
        for edge_jsonld in jsonld_input['edges']:
            self.add_edge(edge=CompetencyOccupationEdge.from_jsonld(edge_jsonld))

    def _build_from_edges(self, edges):
        self._competency_occupation_edges = edges
        self.competency_framework.competencies = [edge.competency for edge in edges]
        self._occupations = dict((edge.occupation.identifier, edge.occupation) for edge in edges)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return \
                self.competencies == other.competencies \
                and self.occupations == other.occupations \
                and self.edges == other.edges


    def __str__(self):
        return f'Competency Framework with {len(self.competencies)} competencies' + \
            f', {len(self.occupations)} occupations' + \
            f', {len(self.edges)} competency-occupation edges'

    @property
    def competencies(self):
        return set(self.competency_framework.values()) - {DummyCompetency()}

    @property
    def occupations(self):
        return set(self._occupations.values()) - {DummyOccupation()}

    @property
    def edges(self):
        values = set([
            edge for edge in self._competency_occupation_edges
            if edge.occupation != DummyOccupation() and edge.competency != DummyCompetency()
        ])
        return values

    def add_competency(self, competency: Competency):
        if not isinstance(competency, Competency):
            raise ValueError('Must add competency objects')
        if competency.identifier not in self.competency_framework:
            self.competency_framework[competency.identifier] = competency
            self.add_edge(competency=competency, occupation=DummyOccupation())
        else:
            self.competency_framework[competency.identifier].children.update(competency.children)
            self.competency_framework[competency.identifier].parents.update(competency.parents)

    def add_occupation(self, occupation: Occupation):
        if not isinstance(occupation, Occupation):
            raise ValueError('Must add occupation objects')
        if occupation.identifier not in self._occupations:
            self._occupations[occupation.identifier] = occupation
            self.add_edge(competency=DummyCompetency(), occupation=occupation)
        else:
            self._occupations[occupation.identifier].children.update(occupation.children)
            self._occupations[occupation.identifier].parents.update(occupation.parents)

    def add_edge(self, occupation: Occupation=None, competency: Competency=None, edge: CompetencyOccupationEdge=None):
        if edge:
            self._competency_occupation_edges.add(edge)
            return

        if not isinstance(occupation, Occupation) or not isinstance(competency, Competency):
            raise ValueError('Must pass both an occupation and competency')
        self.add_competency(competency)
        self.add_occupation(occupation)
        edge = CompetencyOccupationEdge(
            occupation=self._occupations[occupation.identifier],
            competency=self.competency_framework[competency.identifier]
        )
        if edge not in self._competency_occupation_edges:
            self._competency_occupation_edges.add(edge)

    def filter_by(self, func: Callable, competency_name=None, competency_description=None):
        """Produce an ontology that is filtered by a callable that takes in an edge
        """
        matching_edges = set(edge for edge in self._competency_occupation_edges if func(edge))
        return CompetencyOntology(edges=matching_edges, competency_name=competency_name, competency_description=competency_description)

    @property
    def jsonld(self):
        return json.dumps({
            'name': self.name,
            'competencies': [
                competency.jsonld_full
                for competency in
                sorted(self.competencies, key=lambda competency: competency.identifier)
            ],
            'occupations': [
                occupation.jsonld_full
                for occupation in
                sorted(self.occupations, key=lambda occupation: occupation.identifier)
            ],
            'edges': [
                edge.jsonld_full
                for edge in
                sorted(self.edges, key=lambda edge: edge.identifier)
            ]
        }, sort_keys=True)

    def save(self, storage):
        storage.write(self.jsonld.encode('utf-8'), self.name + '.json')

    @property
    def occupation_counts_per_competency(self):
        counts = []
        for competency, edges in itertools.groupby(
            sorted(self._competency_occupation_edges, key=lambda edge: edge.competency),
            lambda edge: edge.competency
        ):
            if competency != DummyCompetency():
                counts.append(len(set([
                    edge.occupation for edge in list(edges) if edge.occupation != DummyOccupation()
                ])))
        return counts

    @property
    def competency_counts_per_occupation(self):
        counts = []
        for occupation, edges in itertools.groupby(
            sorted(self._competency_occupation_edges, key=lambda edge: edge.occupation),
            lambda edge: edge.occupation
        ):
            if occupation != DummyOccupation():
                counts.append(len(set([
                    edge.competency for edge in list(edges)
                    if edge.competency != DummyCompetency()
                ])))
        return counts

    def print_summary_stats(self):
        print(f'Ontology summary statistics for {self.name}')
        print(f'Num competencies: {len(self.competency_framework)}')
        print(f'Num occupations: {len(self.occupations)}')
        print(f'Num competency-occupation edges: {len(self.edges)}')
        print(f'Median occupations per competency: {median(self.occupation_counts_per_competency)}')
        print(f'Median competencies per occupation: {median(self.competency_counts_per_occupation)}')
        print(f'Mean occupations per competency: {mean(self.occupation_counts_per_competency)}')
        print(f'Mean competencies per occupation: {mean(self.competency_counts_per_occupation)}')
