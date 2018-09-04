from itertools import product

from skills_ml.algorithms.skill_extractors import (
    FuzzyMatchSkillExtractor,
    ExactMatchSkillExtractor
)

from skills_ml.ontologies.onet import Onet
from skills_ml.evaluation.skill_extractors import candidate_skills_from_sample, metrics_for_candidate_skills
from skills_ml.evaluation.skill_extraction_metrics import TotalOccurrences, TotalVocabularySize, OntologyCompetencyRecall
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from tests.utils import sample_factory

sample = sample_factory(JobPostingCollectionSample())
skill_extractor_classes = [
    FuzzyMatchSkillExtractor,
    ExactMatchSkillExtractor,
]
print('Building ONET, may take a while to download')
full_onet = Onet()
print('Done building ONET! Now subsetting ONET into K,S,A')
ontologies = [
    full_onet.filter_by(lambda edge: 'Knowledge' in edge.competency.categories, competency_name='onet_knowledge', competency_description='ONET Knowledge'),
    full_onet.filter_by(lambda edge: 'Abilities' in edge.competency.categories, competency_name='onet_ability', competency_description='ONET Ability'),
    full_onet.filter_by(lambda edge: 'Skills' in edge.competency.categories, competency_name='onet_skill', competency_description='ONET Skill')
]

print('Starting skill extraction and evaluation loop')
for skill_extractor_class, ontology in product(skill_extractor_classes, ontologies):
    skill_extractor = skill_extractor_class(ontology.competency_framework)
    print(f'Evaluating skill extractor {skill_extractor.name}')
    candidate_skills = candidate_skills_from_sample(sample, skill_extractor)
    metrics = [
        TotalOccurrences(),
        TotalVocabularySize(),
        OntologyCompetencyRecall(ontology)
    ]

    metrics = metrics_for_candidate_skills(
        candidate_skills=candidate_skills,
        sample=sample,
        metrics=metrics
    )
    print(metrics)
