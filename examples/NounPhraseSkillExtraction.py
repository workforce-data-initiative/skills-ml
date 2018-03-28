# Extracting skills using noun phrase endings
#
# To showcase the noun phrase skill extractor, we download open job postings
# from Virginia Tech's open data portal and run them through the skill extractor.
# In the end, we have the most commonly occurring noun phrases ending in
# 'skill' or 'skills'.
from collections import Counter
import json
import logging
import urllib.request

from skills_ml.algorithms.skill_extractors.noun_phrase_ending import SkillEndingPatternExtractor

logging.basicConfig(level=logging.INFO)

VT_DATASET_URL = 'http://opendata.cs.vt.edu/dataset/ab0abac3-2293-4c9d-8d80-22d450254389/resource/9a810771-d6c9-43a8-93bd-144678cbdd4a/download/openjobs-jobpostings.mar-2016.json'


if __name__ == '__main__':
# VT job postings do not include line breaks, so the bulleted-line filter
# will remove all possible matches. Let's turn it off
    pattern_extractor = SkillEndingPatternExtractor(only_bulleted_lines=False)

    logging.info('Downloading sample Virginia Tech open jobs file')
    response = urllib.request.urlopen(VT_DATASET_URL)
    string = response.read().decode('utf-8')
    logging.info('Download complete')
    lines = string.split('\n')
    logging.info('Found %s job posting lines', len(lines))

    skill_counts = Counter()
    for index, line in enumerate(lines):
        if index % 100 == 0:
            logging.info('On job posting # %s', index)
        try:
            description = json.loads(line)['jobDescription']
            new_skills = pattern_extractor.document_skill_counts(description)
            skill_counts += new_skills
        except ValueError:
            logging.warning('Could not decode JSON')
            continue

    logging.info('100 Most Common Skills in job descriptions: %s', skill_counts.most_common(100))
