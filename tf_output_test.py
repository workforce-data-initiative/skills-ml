from skills_ml.algorithms.skill_extractors.tf import tf_sequence_from_candidate_skills
from skills_ml.algorithms.skill_extractors import SkillEndingPatternExtractor
from skills_ml.job_postings.common_schema import JobPostingCollectionSample, JobPostingCollectionFromS3
from boto.s3.connection import S3Connection
import os
import json


class JobPostingCollectionFromJSONFile(object):
    def __init__(self, filename):
        self.postings = []
        with open(filename) as fd:
            for line in fd:
                self.postings.append(line)
    def __iter__(self):
        for line in self.postings:
            yield json.loads(line)

if __name__ == '__main__':
    job_postings = JobPostingCollectionFromJSONFile('0003.json')
    base_skill_extractor = SkillEndingPatternExtractor(only_bulleted_lines=False)
    unique_chars = set()
    unique_words = set()
    unique_tags = set()
    num_lines = 0
    for job_posting in job_postings:
        print('new posting!')
        candidate_skills = list(base_skill_extractor.candidate_skills(job_posting))
        words_and_tags = tf_sequence_from_candidate_skills(candidate_skills)
        with open('data/words.txt', 'a') as words_fh:
            for job_posting_words in words_and_tags['words']:
                for word in job_posting_words:
                    if word not in unique_words:
                        unique_words.add(word)
                    for char in word:
                        if char not in unique_chars:
                            unique_chars.add(char)
                words_fh.write(' '.join(job_posting_words) + '\n')
                num_lines += 1
        with open('data/tags.txt', 'a') as tags_fh:
            for job_posting_tags in words_and_tags['tags']:
                for tag in job_posting_tags:
                    if tag not in unique_tags:
                        unique_tags.add(tag)
                tags_fh.write(' '.join(job_posting_tags) + '\n')
    with open('data/vocab.words.txt', 'w') as vocab_words_fh:
        for word in unique_words:
            vocab_words_fh.write(word + '\n')
    with open('data/vocab.tags.txt', 'w') as vocab_tags_fh:
        for tag in unique_tags:
            vocab_tags_fh.write(tag + '\n')
    with open('data/vocab.chars.txt', 'w') as vocab_chars_fh:
        for char in unique_chars:
            vocab_chars_fh.write(char + '\n')

    print('num lines = ', num_lines)
    train_range = (0, int(num_lines / 2))
    testa_range = (train_range[1], train_range[1] + int(num_lines/4))
    testb_range = (testa_range[1], num_lines)
    print(train_range)
    print(testa_range)
    print(testb_range)
    with open('data/testa.words.txt', 'w') as testa_words_fh,\
        open('data/testa.tags.txt', 'w') as testa_tags_fh,\
        open('data/testb.words.txt', 'w') as testb_words_fh,\
        open('data/testb.tags.txt', 'w') as testb_tags_fh,\
        open('data/train.words.txt', 'w') as train_words_fh,\
        open('data/train.tags.txt', 'w') as train_tags_fh:
        with open('data/words.txt') as words_fh, open('data/tags.txt') as tags_fh:
            for lineno, (wordline, tagline) in enumerate(zip(words_fh, tags_fh)):
                if lineno < train_range[1]:
                    train_words_fh.write(wordline.rstrip() + '\n')
                    train_tags_fh.write(tagline.rstrip() + '\n')
                elif lineno < testa_range[1]:
                    testa_words_fh.write(wordline.rstrip() + '\n')
                    testa_tags_fh.write(tagline.rstrip() + '\n')
                else:
                    testb_words_fh.write(wordline.rstrip() + '\n')
                    testb_tags_fh.write(tagline.rstrip() + '\n')
    os.unlink('data/words.txt')
    os.unlink('data/tags.txt')
