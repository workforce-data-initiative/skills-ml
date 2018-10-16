from .base import Competency, Occupation, CompetencyOntology
from .clustering import Clustering, jaccard_competency_distance
from skills_ml.datasets.onet_cache import OnetSiteCache
from descriptors import cachedproperty
from collections import defaultdict
import logging
from typing import Callable
from scipy.spatial.distance import pdist, squareform
import numpy as np

majorgroupname = {
    '11': 'Management Occupations',
    '13': 'Business and Financial Operations Occupations',
    '15': 'Computer and Mathematical Occupations',
    '17': 'Architecture and Engineering Occupations',
    '19': 'Life, Physical, and Social Science Occupations',
    '21': 'Community and Social Service Occupations',
    '23': 'Legal Occupations',
    '25': 'Education, Training, and Library Occupations',
    '27': 'Arts, Design, Entertainment, Sports, and Media Occupations',
    '29': 'Healthcare Practitioners and Technical Occupations',
    '31': 'Healthcare Support Occupations',
    '33': 'Protective Service Occupations',
    '35': 'Food Preparation and Serving Related Occupations',
    '37': 'Building and Grounds Cleaning and Maintenance',
    '39': 'Personal Care and Service Occupations',
    '41': 'Sales and Related Occupations',
    '43': 'Office and Administrative Support Occupations',
    '45': 'Farming, Fishing, and Forestry Occupations',
    '47': 'Construction and Extraction Occupations',
    '49': 'Installation, Maintenance, and Repair Occupations',
    '51': 'Production Occupations',
    '53': 'Transportation and Material Moving Occupations',
    '55': 'Military Specific Occupations'
}


class Onet(CompetencyOntology):
    def __init__(self, onet_cache=None, manual_build=True):
        if manual_build:
            logging.info('Manual build specified. Building O*NET CompetencyOntology via direct querying from O*NET site, or local cache.')
            super().__init__()
            self.is_built = False
            self.onet_cache = onet_cache or OnetSiteCache()
            self.name = 'onet'
            self.competency_framework.name = 'onet_ksat'
            self.competency_framework.description = 'ONET Knowledge, Skills, Abilities, Tools, and Technology'
            self._build()
        else:
            logging.info('Building O*NET CompetencyOntology via Research Hub-hosted JSON-LD')
            super().__init__(research_hub_name='onet')

    def _build(self):
        if not self.is_built:
            onet_cache = self.onet_cache
            description_lookup = {}
            logging.info('Processing Content Model Reference')
            for row in onet_cache.reader('Content Model Reference'):
                description_lookup[row['Element ID']] = row['Description']

            logging.info('Processing occupation data')
            for row in onet_cache.reader('Occupation Data'):
                occupation = Occupation(
                        identifier=row['O*NET-SOC Code'],
                        name=row['Title'],
                        description=row['Description'],
                        categories=['O*NET-SOC Occupation'],
                )
                major_group_num = row['O*NET-SOC Code'][0:2]
                major_group = Occupation(
                    identifier=major_group_num,
                    name=majorgroupname[major_group_num],
                    categories=['O*NET-SOC Major Group']
                )
                occupation.add_parent(major_group)
                self.add_occupation(occupation)
                self.add_occupation(major_group)

            logging.info('Processing Knowledge, Skills, Abilities')
            for content_model_file in {'Knowledge', 'Abilities', 'Skills'}:
                for row in onet_cache.reader(content_model_file):
                    if row['Scale ID'] == 'IM' and float(row['Data Value']) >= 3:
                        competency = Competency(
                            identifier=row['Element ID'],
                            name=row['Element Name'],
                            categories=[content_model_file],
                            competencyText=description_lookup[row['Element ID']]
                        )
                        self.add_competency(competency)
                        occupation = Occupation(identifier=row['O*NET-SOC Code'])
                        self.add_edge(competency=competency, occupation=occupation)

            logging.info('Processing tools and technology')
            for row in onet_cache.reader('Tools and Technology'):
                key = row['Commodity Code'] + '-' + row['T2 Example']
                commodity_competency = Competency(
                    identifier=row['Commodity Code'],
                    name=row['Commodity Title'],
                    categories=[row['T2 Type'], 'UNSPSC Commodity'],
                )
                competency = Competency(
                    identifier=key,
                    name=row['T2 Example'],
                    categories=[row['T2 Type'], 'O*NET T2'],
                )
                competency.add_parent(commodity_competency)
                self.add_competency(commodity_competency)
                self.add_competency(competency)
                occupation = Occupation(identifier=row['O*NET-SOC Code'])
                self.add_edge(competency=competency, occupation=occupation)

            self.is_built = True
        else:
            logging.warning('O*Net Ontology is already built!')

    @cachedproperty
    def all_soc(self):
        occupations = self.occupations
        soc = []
        for occ in occupations:
            if 'O*NET-SOC Occupation' in occ.other_attributes['categories']:
                soc.append(occ.identifier)
        return sorted(soc)

    @cachedproperty
    def all_major_groups(self):
        occupations = self.occupations
        major_groups = []
        for occ in occupations:
            if 'O*NET-SOC Major Group' in occ.other_attributes['categories']:
                major_groups.append(occ)
        return sorted(major_groups, key=lambda k: k.identifier)

    @cachedproperty
    def all_major_groups_occ(self):
        occ = self.filter_by(lambda edge: len(edge.occupation.identifier) == 2)
        return occ.occupations

    @cachedproperty
    def competency_categories(self):
        return set(c.categories[0] for c in self.competencies)

    @cachedproperty
    def major_group_occupation_name_clustering(self):
        d = Clustering(
                name="major_group_occupations_name",
                key_transform_fn=lambda concept: getattr(concept, "name"),
                value_transform_fn=lambda entity: getattr(entity, "name"),
        )
        for mg in self.all_major_groups_occ:
            d[mg] = [child for child in mg.children]
        d.transform()
        return d

    @cachedproperty
    def major_group_occupation_description_clustering(self):
        d = Clustering(
                name="major_group_occupations_description",
                key_transform_fn=lambda concept: getattr(concept, "name"),
                value_transform_fn=lambda entity: ' '.join([
                    str(getattr(entity, "name")),
                    str(getattr(entity, "other_attributes").get("description"))]),
                )
        for mg in self.all_major_groups_occ:
            d[mg] = [child for child in mg.children]
        d.transform()
        return d

    @cachedproperty
    def major_group_competencies_name_clustering(self):
        d = Clustering(
                name="major_group_competencies_name",
                key_transform_fn=lambda concept: getattr(concept, "name"),
                value_transform_fn=lambda entity: getattr(entity, "name"),
        )
        for mg in self.all_major_groups_occ:
            d[mg] = self.filter_by(lambda edge: edge.occupation.identifier[:2] == mg.identifier[:2]).competencies
        d.transform()
        return d

    @cachedproperty
    def major_group_competencies_description_cluster(self):
        d = Clustering(
                name="major_group_competencies_description",
                key_transform_fn=lambda concept: getattr(concept, "name"),
                value_transform_fn=lambda entity: ' '.join([
                    str(getattr(entity, "name")),
                    str(getattr(entity, "other_attributes").get("competencyText"))]),
        )
        for mg in self.all_major_groups_occ:
            d[mg] = self.filter_by(lambda edge: edge.occupation.identifier[:2] == mg.identifier[:2]).competencies
        d.transform()
        return d

    def occupation_competency_dict(self, key_fn: Callable=lambda occ_id: True):
        occ_dict = defaultdict(list)
        for edge in self._competency_occupation_edges:
            if key_fn(edge.occupation.identifier):
                occ_dict[edge.occupation.identifier].append(edge.competency)
        return occ_dict

    def distance_matrix_for_occupation(self, occupation_dict=None):
        if occupation_dict is None:
            occupation_dict = self.occupation_competency_dict(lambda occ_id: len(occ_id) > 2)
        occ_keys = np.array(list(occupation_dict.keys()))
        occ_values = np.array(list(occupation_dict.values()))
        occ_values = occ_values.reshape(len(occupation_dict), 1)

        d_matrix = pdist(occ_values, lambda u, v: jaccard_competency_distance(u[0], v[0], self.competency_categories))

        return occ_keys, squareform(d_matrix)

    def generate_clusterings(self):
        return [self.major_group_occupation_clustering, self.major_group_competencies_clustering]
