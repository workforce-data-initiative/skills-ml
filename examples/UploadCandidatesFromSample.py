import logging

from itertools import product
from functools import partial
from multiprocessing import Pool

from skills_ml.algorithms.sampling import Sample
from skills_ml.algorithms.skill_extractors.freetext import \
    FuzzyMatchSkillExtractor,\
    ExactMatchSkillExtractor,\
    OccupationScopedSkillExtractor
from skills_ml.algorithms.skill_extractors import upload_candidates_from_sample


def generate_skill_candidates(candidates_path, sample, skill_extractor, n_jobs):
    pool = Pool(n_jobs)
    for result in pool.imap(
        partial(upload_candidates_from_sample, candidates_path, skill_extractor, sample.name),
        sample
    ):
        logging.info(result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    n_jobs = 3
    #sample_names = ['samples_24k_v1', 'samples_10k_v1', 'samples_300_v1']
    sample_names = ['samples_300_v1']
    skill_extractor_classes = [
        FuzzyMatchSkillExtractor,
        ExactMatchSkillExtractor,
        #OccupationScopedSkillExtractor
    ]
    sample_path = 's3://open-skills-private/sampled_jobpostings'
    candidates_path = 'open-skills-private/skill_candidates'
    skills_master_table_path = 
    skill_tables = [
        #('s3://open-skills-public/pipeline/tables/skills_master_table.tsv', 'onet_ksat'),
        ('s3://open-skills-public/skill_lists/onet_knowledge.tsv', 'onet_knowledge'),
        ('s3://open-skills-public/skill_lists/onet_skill.tsv', 'onet_skill'),
        ('s3://open-skills-public/skill_lists/onet_ability.tsv', 'onet_ability'),
        #('s3://open-skills-public/skill_lists/onet_tools_tech.tsv', 'onet_tools_tech'),
    ]
    for sample_name, skill_extractor_class, skill_table in product(sample_names, skill_extractor_classes, skill_tables):
        sample = Sample(sample_path, sample_name)
        skill_extractor = skill_extractor_class(
            skill_lookup_path=skill_table[0],
            skill_lookup_type=skill_table[1]
        )
    generate_skill_candidates(candidates_path, sample, skill_extractor, n_jobs)
