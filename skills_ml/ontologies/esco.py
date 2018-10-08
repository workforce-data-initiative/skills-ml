from .base import Competency, Occupation, CompetencyOntology
import logging
import csv
import requests

lang = '&language=en'
api_tax = 'https://ec.europa.eu/esco/api/resource/taxonomy?uri=http://data.europa.eu/esco/concept-scheme/isco&language=en'


class Esco(CompetencyOntology):
    name = 'esco'

    def __init__(self, manual_build=False):
        if manual_build:
            logging.info('Manual build specified. Building ESCO CompetencyOntology via direct querying from ESCO. Beware, this could take hours!')
            super().__init__()
            self.is_built = False
            self.competency_framework.name = 'esco_skills'
            self.competency_framework.description = 'ESCO Knowledge, Skills/Competencies'
            self._build()
            self.is_built = True
        else:
            # by default, build from research hub
            logging.info('Building ESCO CompetencyOntology via Research Hub-hosted JSON-LD')
            super().__init__(research_hub_name='esco')

    
    def _build(self):
        if not self.is_built:
            reqTax = requests.get(api_tax)
            getConcepts = reqTax.json()['_links']['hasTopConcept']
            num_concepts = len(getConcepts)
            for i, concept in enumerate(getConcepts):
                logging.info('Processing concept %s of %s: %s', i, num_concepts, concept['href'])
                self._conceptProcessing(concept['href'])
        else:
            logging.warning('ESCO Ontology is already built!')

    def _conceptProcessing(self, href):
        reqOcc = requests.get(href)
        links = reqOcc.json()['_links']
        if 'narrowerOccupation' in links:
            self._occupationProcessing(links['narrowerOccupation'])
        elif 'narrowerConcept' in links:
            _concepts = links['narrowerConcept']
            for _concept in _concepts:
                self._conceptProcessing(_concept['href'])

    def getOrCreateCompetency(self, skill):
        competency = self.competency_framework.competencies.get(skill['uri'], None)
        if not competency:
            logging.info('Competency %s not found, getting data', skill['uri'])
            competency = self._addCompetency(skill['href'], skill['uri'], skill['title'])
        else:
            logging.info('Competency %s found, no need to get data!', skill['uri'])
        return competency

    def _addCompetency(self, href, uri, title):
        reqSkill = requests.get(href).json()
        competency = Competency(
            identifier=uri,
            name=title,
            categories=reqSkill['_links']['hasSkillType'][0]['title'],
            competencyText=reqSkill['description']['en']['literal']
        )
        self.add_competency(competency)
        for skill in reqSkill['_links'].get('broaderSkill', []):
            logging.info('Processing broader skill %s', skill)
            broaderCompetency = self.getOrCreateCompetency(skill)
            broaderCompetency.add_child(competency)
            self.add_competency(broaderCompetency)
        for skill in reqSkill['_links'].get('narrowerSkill', []):
            logging.info('Processing narrower skill %s', skill)
            narrowerCompetency = self.getOrCreateCompetency(skill)
            narrowerCompetency.add_parent(competency)
            self.add_competency(narrowerCompetency)
        return competency

    def _occupationProcessing(self, _occupations):
        logging.info('Processing Occupations')
        for occ in _occupations:
            reqOcc = requests.get(occ['href'])
            reqOccJSON = reqOcc.json()
            occupation = Occupation(
                identifier=occ['uri'],
                name=occ['title'],
                description=reqOccJSON['description']['en']['literal'],
                categories='ESCO Occupation')
            iscoGrps = reqOccJSON['_links']['broaderIscoGroup']
            self.add_occupation(occupation)
            self._parentProcessing(iscoGrps, occupation)

            logging.info('Processing Skills/Competencies & Knowledge for the %s occupation', occ['title'])
            essentialSkills = reqOccJSON['_links'].get('hasEssentialSkill', [])
            optionalSkills = reqOccJSON['_links'].get('hasOptionalSkill', [])
            allSkills = essentialSkills + optionalSkills
            for skill in allSkills:
                competency = self.getOrCreateCompetency(skill)
                self.add_edge(competency=competency, occupation=occupation)

    def _parentProcessing (self, concepts, occupation):
        for iscoGrp in concepts:
            major_group = Occupation(
                identifier=iscoGrp['uri'],
                name=iscoGrp['title'],
                categories='ESCO Concept')
            occupation.add_parent(major_group)
            self.add_occupation(major_group)
            if 'broaderConcept' in iscoGrp:
                self._parentProcessing (iscoGrp['broaderConcept'], major_group)
