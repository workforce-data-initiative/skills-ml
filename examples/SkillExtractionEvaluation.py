from skills_ml.algorithms.skill_extractors import (
    FuzzyMatchSkillExtractor,
    ExactMatchSkillExtractor,
    SocScopedExactMatchSkillExtractor,
    SectionExtractSkillExtractor,
    SkillEndingPatternExtractor,
    AbilityEndingPatternExtractor
)

from skills_ml.ontologies.onet import Onet
from skills_ml.evaluation.skill_extractors import candidate_skills_from_sample, metrics_for_candidate_skills
from skills_ml.evaluation.skill_extraction_metrics import TotalOccurrences, TotalVocabularySize, OntologyCompetencyRecall
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from tests.utils import sample_factory

sample = sample_factory(JobPostingCollectionSample())
print('Building ONET, may take a while to download')
full_onet = Onet()

skill_extractors = [
    SectionExtractSkillExtractor(),
    SkillEndingPatternExtractor(only_bulleted_lines=False),
    AbilityEndingPatternExtractor(only_bulleted_lines=False),
    FuzzyMatchSkillExtractor(full_onet.competency_framework),
    ExactMatchSkillExtractor(full_onet.competency_framework),
    SocScopedExactMatchSkillExtractor(full_onet)
]
print('Done building ONET! Now subsetting ONET into K,S,A')
metric_ontologies = [
    full_onet,
    full_onet.filter_by(lambda edge: 'Knowledge' in edge.competency.categories, competency_name='onet_knowledge', competency_description='ONET Knowledge'),
    full_onet.filter_by(lambda edge: 'Abilities' in edge.competency.categories, competency_name='onet_ability', competency_description='ONET Ability'),
    full_onet.filter_by(lambda edge: 'Skills' in edge.competency.categories, competency_name='onet_skill', competency_description='ONET Skill')
]
metrics = [
    TotalOccurrences(),
    TotalVocabularySize(),
]
for metric_ontology in metric_ontologies:
    metrics.append(OntologyCompetencyRecall(metric_ontology))

print('Starting skill extraction and evaluation loop')
for skill_extractor in skill_extractors:
    print(f'Evaluating skill extractor {skill_extractor.name}')
    candidate_skills = candidate_skills_from_sample(sample, skill_extractor)

    computed_metrics = metrics_for_candidate_skills(
        candidate_skills=candidate_skills,
        sample=sample,
        metrics=metrics
    )
    print(computed_metrics)
