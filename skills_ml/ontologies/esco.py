#from .base import Competency, Occupation, CompetencyOntology
import logging
import csv
import requests

majorgroupname = {
    '01': 'Commissioned Armed Forces Officers',
    '02': 'Non-commissioned Armed Forces Officers',
    '03': 'Armed Forces Occupations, Other Ranks',
    '11': 'Chief Executives, Senior Officials and Legislators',
    '12': 'Administrative and Commercial Managers',
    '13': 'Production and Specialized Services Managers',
    '14': 'Hospitality, Retail and Other Services Managers',
    '21': 'Science and Engineering Professionals',
    '22': 'Health Professionals',
    '23': 'Teaching Professionals',
    '24': 'Business and Administration Professionals',
    '25': 'Information and Communications Technology Professionals',
    '26': 'Legal, Social and Cultural Professionals',
    '31': 'Science and Engineering Associate Professionals',
    '32': 'Health Associate Professionals', 
    '33': 'Business and Administration Associate Professionals',
    '34': 'Legal, Social, Cultural and Related Associate Professionals',
    '35': 'Information and Communications Technicians',
    '41': 'General and Keyboard Clerks',
    '42': 'Customer Services Clerks',
    '43': 'Numerical and Material Recording Clerks',
    '44': 'Other Clerical Support Workers',
    '51': 'Personal Services Workers',
    '52': 'Sales Workers',
    '53': 'Personal Care Workers',
    '54': 'Protective Services Workers',
    '61': 'Market-oriented Skilled Agricultural Workers',
    '62': 'Market-oriented Skilled Forestry, Fishery and Hunting Workers',
    '63': 'Subsistence Farmers, Fishers, Hunters and Gatherers',
    '71': 'Building and Related Trades Workers (excluding electricians)',
    '72': 'Metal, Machinery and Related Trades Workers',
    '73': 'Handicraft and Printing Workers',
    '74': 'Electrical and Electronics Trades Workers',
    '75': 'Food Processing, Woodworking, Garment and Other Craft and Related Trades Workers',
    '81': 'Stationary Plant and Machine Operators',
    '82': 'Assemblers',
    '83': 'Drivers and Mobile Plant Operators',
    '91': 'Cleaners and Helpers',
    '92': 'Agricultural, Forestry and Fishery Labourers',
    '93': 'Labourers in Mining, Construction, Manufacturing and Transport',
    '94': 'Food Preparation Assistants',
    '95': 'Street and Related Sales and Services Workers',
    '96': 'Refuse Workers and Other Elementary Workers'
}
api_occ = 'https://ec.europa.eu/esco/api/resource/occupation?uri='
lang = '&language=en'
api_skills = 'https://ec.europa.eu/esco/api/resource/skill?uri='

class Esco(CompetencyOntology):
     def __init__(self):
        super().__init__()
        self.is_built = False
        self.competency_framework.name = 'esco_ksat'
        self.competency_framework.description = 'ESCO Knowledge, Skills/Competencies'
        self._build()
    
     def _build(self):
        if not self.is_built:
            logging.info('Processing occupation data')
            with open('esco/occupations_en.csv') as occupations_csv:
                reader = csv.DictReader(occupations_csv)
                for row in reader:
                    uri = row['conceptUri']
                    occupation = Occupation(
                        identifier=row['iscoGroup'],
                        name=row['preferredLabel'],
                        description=row['description'],
                        categories=['ESCO Occupation'] 
                    )
                    major_group_num = row['iscoGroup'][0:2]
                    major_group = Occupation(
                        identifier=major_group_num,
                        name=majorgroupname[major_group_num],
                        categories=['ESCO Occupation'] 
                    )
                    occupation.add_parent(major_group)
                    self.add_occupation(occupation)
                    self.add_occupation(major_group)
                    
                    logging.info('Processing Skills/Competencies & Knowledge')
                    reqOcc = requests.get(api_occ + uri + lang)
                    essentialSkills = reqOcc.json()['_links']['hasEssentialSkill']
                    for essSkill in essentialSkills:
                        reqSkill = requests.get(api_skills + essSkill['uri'] + lang)
                        inSkill = reqSkill.json()['_links']['hasSkillType']
                        competency = Competency(
                            identifier=essSkill['uri']
                            name=essSkill['title']
                            categories=inSkill[0]['title']
                            competencyText=reqSkill.json()['description']['en']['literal']
                        )
                    self.add_competency(competency)
                    occupation = Occupation(identifier=row['iscoGroup'])
                    self.add_edge(competency=competency, occupation=occupation)
            
                self.is_built = True
                occupations_csv.close()
        else:
            logging.warning('ESCO Ontology is already built!')
            
                
                
            