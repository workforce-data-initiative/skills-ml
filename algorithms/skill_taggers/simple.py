import csv
import logging

from utils.nlp import NLPTransforms

from algorithms.skill_taggers.base import SkillTaggerBase


class SimpleSkillTagger(SkillTaggerBase):
    def __init__(self, hash_function, *args, **kwargs):
        super(SimpleSkillTagger, self).__init__(*args, **kwargs)
        self.tracker = {
            'total_skills': 0,
            'jobs_with_skills': 0
        }
        self.nlp = NLPTransforms()
        self.hash_function = hash_function
        self.lookup = self._skills_lookup()
        logging.info(
            'Done creating skills lookup with %d entries',
            len(self.lookup)
        )

    def _skills_lookup(self):
        logging.info('Creating skills lookup from %s', self.skills_filename)
        with open(self.skills_filename) as infile:
            reader = csv.reader(infile, delimiter='\t')
            header = next(reader)
            index = header.index(self.nlp.transforms[0])
            generator = (row[index] for row in reader)
            return set(generator)

    def labeler(self, document):
        def label(document, lookup, tracker):
            join_spaces = " ".join  # for runtime efficiency
            found_skills = 0

            N = 5
            doc = document.split()
            doc_len = len(doc)

            start_idx = 0

            # Yield a generator of document skills/non skills that advances
            # index pointer (`offset`) to end of skill or current non skill
            # while yielding that skill uuid or non skill token (controlled by
            # `found_skill` flag) to the callee so they can do whatever with it
            while start_idx < doc_len:
                found_skill = False
                offset = 1

                lookahead = min(N, doc_len - start_idx)
                for idx in range(lookahead, 0, -1):
                    ngram = join_spaces(doc[start_idx:start_idx+idx])
                    if ngram in lookup:
                        found_skill = True
                        offset = idx
                        found_skills += 1
                        yield self.hash_function(ngram)
                        break

                if not found_skill:
                    yield doc[start_idx]

                start_idx += offset
            if found_skills > 0:
                tracker['jobs_with_skills'] += 1
                tracker['total_skills'] += found_skills

        return ' '.join(list(label(document, self.lookup, self.tracker)))

    def _label_titles(self, corpus_generator):
        for corpus in corpus_generator:
            yield self.labeler(corpus)
        logging.info(
            'Done labeling titles, tracker results: %s',
            self.tracker
        )


if __name__ == '__main__':
    tagger = SimpleSkillTagger(
        raw_filename='raw_corpora_a.txt',
        labeled_filename='labeled_corpora_a.tsv',
        skills_filename='skills_master_table.tsv'
    )
    tagger.run()
