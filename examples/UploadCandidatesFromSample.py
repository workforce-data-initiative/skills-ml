import logging

from itertools import product
from functools import partial
from multiprocessing import Pool

from skills_ml.algorithms.sampling import Sample
from skills_ml.algorithms.skill_extractors.freetext import \
    FuzzyMatchSkillExtractor,\
    ExactMatchSkillExtractor,\
    OccupationScopedSkillExtractor
from skills_ml.algorithms.skill_extractors import upload_candidates_from_job_posting_json


def generate_skill_candidates_multiprocess(candidates_path, sample, skill_extractor, n_jobs):
    pool = Pool(n_jobs)
    for result in pool.imap(
        partial(upload_candidates_from_job_posting_json, candidates_path, skill_extractor, sample.name),
        sample
    ):
        logging.info(result)

def generate_skill_candidates_oneprocess(candidates_path, sample, skill_extractor):
    for job_posting_json in sample:
        upload_candidates_from_job_posting_json(candidates_path, skill_extractor, job_posting_json, sample.name)

PRIVATE_BUCKET = 'sample-private'
PUBLIC_BUCKET = 'sample-public'

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    n_jobs = 3
    sample_names = ['samples_300_v1', 'samples_10k_v1']
    skill_extractor_classes = [
        FuzzyMatchSkillExtractor,
        ExactMatchSkillExtractor,
    ]
    sample_path = 's3://{}/sampled_jobpostings'.format(PRIVATE_BUCKET)
    candidates_path = '{}/skill_candidates'.format(PRIVATE_BUCKET)
    skill_tables = [
        ('s3://{}/skill_lists/onet_knowledge.tsv'.format(PUBLIC_BUCKET), 'onet_knowledge'),
        ('s3://{}/skill_lists/onet_skill.tsv'.format(PUBLIC_BUCKET), 'onet_skill'),
        ('s3://{}/skill_lists/onet_ability.tsv'.format(PUBLIC_BUCKET), 'onet_ability'),
    ]
    for sample_name, skill_extractor_class, skill_table in product(sample_names, skill_extractor_classes, skill_tables):
        sample = Sample(sample_path, sample_name)
        skill_extractor = skill_extractor_class(
            skill_lookup_path=skill_table[0],
            skill_lookup_type=skill_table[1]
        )
        generate_skill_candidates_oneprocess(candidates_path, sample, skill_extractor)
