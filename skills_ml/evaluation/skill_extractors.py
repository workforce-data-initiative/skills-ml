import json
import logging
from skills_ml.algorithms.sampling import Sample
from skills_ml.algorithms.skill_extractors.base import SkillExtractor, CandidateSkill
from skills_ml.storage import Store
from skills_ml.evaluation.skill_extraction_metrics import SkillExtractorMetric


CANDIDATE_SKILL_FILENAME = 'candidate_skills.json'
METRICS_FILENAME = 'metrics.json'


def candidate_skills_from_sample(
    sample: Sample,
    skill_extractor: SkillExtractor,
    output_storage: Store=None
):
    all_candidate_skills = []
    for line in sample:
        all_candidate_skills.extend(
            skill_extractor.candidate_skills(json.loads(line))
        )
    if output_storage:
        output_storage.write(
            json.dumps([cs._asdict() for cs in all_candidate_skills]).encode('utf-8'),
            f'{sample.name}/{CANDIDATE_SKILL_FILENAME}'
        )
    return all_candidate_skills


def metrics_for_candidate_skills(
    sample,
    metrics,
    candidate_skills=None,
    input_storage=None,
    output_storage=None
):
    if candidate_skills is None:
        if not input_storage:
            raise ValueError('Either a list of CandidateSkills or an input_storage that points to a collection of CandidateSkills is needed')
        stored_objects = json.loads(input_storage.load(f'{sample.name}/{CANDIDATE_SKILL_FILENAME}'))
        candidate_skills = []
        for stored_object in stored_objects:
            candidate_skills.append(CandidateSkill(**stored_object))
        logging.info('Successfully loaded candidate skills from storage')
    logging.info('Processing %s candidate skills', len(candidate_skills))
    computed_metrics = {}
    for metric in metrics:
        computed_metrics[metric.name] = metric.eval(candidate_skills, len(sample))
    if output_storage:
        output_storage.write(
            json.dumps(computed_metrics).encode('utf-8'),
            f'{sample.name}/{METRICS_FILENAME}'
        )
    return computed_metrics
