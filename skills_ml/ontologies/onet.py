from .base import Competency, Occupation, CompetencyOntology
from skills_ml.datasets.onet_cache import OnetSiteCache
import logging


def build_onet(onet_cache=None):
    if not onet_cache:
        onet_cache = OnetSiteCache()

    ontology = CompetencyOntology()
    description_lookup = {}
    logging.info('Processing Content Model Reference')
    for row in onet_cache.reader('Content Model Reference'):
        description_lookup[row['Element ID']] = row['Description']

    logging.info('Processing occupation data')
    for row in onet_cache.reader('Occupation Data'):
        occupation = Occupation(
            identifier=row['O*NET-SOC Code'],
            name=row['Title'],
            description=row['Description']
        )
        major_group_num = row['O*NET-SOC Code'][0:2]
        major_group = Occupation(
            identifier=major_group_num,
            name=f'Major Group {major_group_num}'
        )
        occupation.add_parent(major_group)
        ontology.add_occupation(occupation)
        ontology.add_occupation(major_group)

    logging.info('Processing Knowledge, Skills, Abilities')
    for content_model_file in {'Knowledge', 'Abilities', 'Skills'}:
        for row in onet_cache.reader(content_model_file):
            competency = Competency(
                identifier=row['Element ID'],
                name=row['Element Name'],
                category=content_model_file,
                competencyText=description_lookup[row['Element ID']]
            )
            ontology.add_competency(competency)
            occupation = Occupation(identifier=row['O*NET-SOC Code'])
            ontology.add_edge(competency=competency, occupation=occupation)

    logging.info('Processing tools and technology')
    for row in onet_cache.reader('Tools and Technology'):
        key = row['Commodity Code'] + '-' + row['T2 Example']
        commodity_competency = Competency(
            identifier=row['Commodity Code'],
            name=row['Commodity Title'],
            category=row['T2 Type'],
        )
        competency = Competency(
            identifier=key,
            name=row['T2 Example'],
            category=row['T2 Type'],
        )
        competency.add_parent(commodity_competency)
        ontology.add_competency(commodity_competency)
        ontology.add_competency(competency)
        occupation = Occupation(identifier=row['O*NET-SOC Code'])
        ontology.add_edge(competency=competency, occupation=occupation)

    return ontology
