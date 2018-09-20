from .base import Competency, Occupation, CompetencyOntology
import logging
import csv
import requests

lang = '&language=en'
api_tax = 'https://ec.europa.eu/esco/api/resource/taxonomy?uri=http://data.europa.eu/esco/concept-scheme/isco&language=en'


class Esco(CompetencyOntology):
    def __init__(self):
        super().__init__()
        self.is_built = False
        self.competency_framework.name = 'esco_skills'
        self.competency_framework.description = 'ESCO Knowledge, Skills/Competencies'
        self._build()
    
    def _build(self):
        if not self.is_built:
            reqTax = requests.get(api_tax)
            getConcepts = reqTax.json()['_links']['hasTopConcept']
            for concept in getConcepts:
                self._conceptProcessing(concept['href'])
        else:
            logging.warning('ESCO Ontology is already built!')

    def _conceptProcessing(self, href):
        reqOcc = requests.get(href)
        if 'narrowerOccupation' in reqOcc.json()['_links']:
            self._occupationProcessing(reqOcc.json()['_links']['narrowerOccupation'])
        elif 'narrowerConcept' in reqOcc.json()['_links']:
            _concepts = reqOcc.json()['_links']['narrowerConcept']
            for _concept in _concepts:
                self._conceptProcessing(_concept['href'])

    def _occupationProcessing(self, _occupations):
        logging.info('Processing Occupations')
        for occ in _occupations:
            reqOcc = requests.get(occ['href'])
            occupation = Occupation(
                identifier=occ['uri'],
                name=occ['title'],
                description=reqOcc.json()['description']['en']['literal'],
                categories='ESCO Occupation')
            iscoGrps = reqOcc.json()['_links']['broaderIscoGroup']
            self.add_occupation(occupation)
            self._parentProcessing(iscoGrps, occupation)

            logging.info('Processing corresponding Skills/Competencies & Knowledge')
            if 'hasEssentialSkill' in reqOcc.json()['_links']:
                essentialSkills = reqOcc.json()['_links']['hasEssentialSkill']
            if 'hasOptionalSkill' in reqOcc.json()['_links']:
                optionalSkills = reqOcc.json()['_links']['hasOptionalSkill']
            allSkills = essentialSkills + optionalSkills
            for skill in allSkills:
                reqSkill = requests.get(skill['href'])
                inSkill = reqSkill.json()['_links']['hasSkillType']
                competency = Competency(
                    identifier=skill['uri'],
                    name=skill['title'],
                    categories=inSkill[0]['title'],
                    competencyText=reqSkill.json()['description']['en']['literal'])
                #print "Skill needed: " + skill['title']
                self.add_competency(competency)
                occupation = Occupation(identifier=skill['uri'])
                self.add_edge(competency=competency, occupation=occupation)
                self.is_built = True

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