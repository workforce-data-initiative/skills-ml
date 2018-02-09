# coding: utf-8
import logging
import re
from collections import Counter

import nltk
try:
    nltk.pos_tag('test')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from .base import SkillExtractor


def sentences_words_pos(document):
    """This function takes raw text and chops and then connects the process to break
       it down into sentences, then words and then complete part-of-speech tagging"""
    try:
        sentences = nltk.sent_tokenize(document.encode('utf-8'))
    except:
        sentences = nltk.sent_tokenize(document)
    sentences = [nltk.wordpunct_tokenize(sent) for sent in sentences]

    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


def count_noun_phrases_in_line(line, counts):
    if line.strip() != "\n":
        output = sentences_words_pos(line)

        grammar = r"""
                  NP: {<JJ.*>*<NN.*><NN.*>*}   # chunk adjectives and nouns
                   """
        cp = nltk.RegexpParser(grammar)
        for sent in output:
            if sent:
                tree = cp.parse(sent)
                for subtree in tree.subtrees():
                    if subtree.label() == 'NP':
                        np = ""
                        for node in subtree:
                            try:
                                np += node[0].encode('utf-8') + ' '
                            except:
                                np += node[0] + ' '
                        np = np.strip()

                        logging.info('Adding %s', np)
                        counts[np] += 1


BULLETS = ['+', '*', '-']


def is_bulleted(string):
    string_first = string.split(' ', 1)[0].replace('\n', '')
    try:
        if string_first[0] in BULLETS:
            return True
        else:
            return False
    except:
        return False


BEGINNING_CHARS_TO_REMOVE = [
    "+ ",
    "â €¢ ",
    "Â · ",
    "Â · Â Â Â Â Â Â Â ",
    "\. ",
    "\- ",
    "/ ",
    "**,** ",
    "** ",
    "* ",
    "á ",
    "### ",
    "‰ Û¢ ",
    "å á å å å å å å å ",
    "å á ",
    "á "
]


def clean_beginning(term):
    for c in BEGINNING_CHARS_TO_REMOVE:
        if term.startswith(c):
            term = term.replace(c, "")
            break
    return term


class NPEndPatternExtractor(SkillExtractor):
    def __init__(self, endings, stop_phrases, only_bulleted_lines=True, *args, **kwargs):
        self.endings = endings
        self.stop_phrases = stop_phrases
        self.only_bulleted_lines = only_bulleted_lines

    def document_skill_counts(self, document):
        """Count skills in the document

        Args:
            document (string) A document for searching, such as a job posting

        Returns: (collections.Counter) skills found in the document, all
            values set to 1 (multiple occurrences of a skill do not count)
        """
        noun_phrase_counts = Counter()
        lines = re.findall(r'\n.*', document)
        for line in lines:
            if not self.only_bulleted_lines or is_bulleted(line):
                count_noun_phrases_in_line(line, noun_phrase_counts)
        skill_counts = Counter()
        for phrase, frequency in noun_phrase_counts.items():
            term_list = phrase.split()
            if term_list[-1].lower() in self.endings:
                    cleaned_phrase = clean_beginning(phrase).lower()
                    if cleaned_phrase not in self.stop_phrases:
                        skill_counts[cleaned_phrase] += frequency
        return skill_counts


class SkillPatternExtractor(NPEndPatternExtractor):
    GENERIC_SKILL_PHRASES = set([
        'advanced skills',
        'basic skills',
        'bonus skills',
        'career skills',
        'core skills',
        'demonstrable skill',
        'demonstrated skill',
        'demonstrated skills',
        'desired additional skills',
        'desired skills',
        'encouraged skills',
        'essential skills',
        'following requisite skills',
        'following skill',
        'following skills',
        'individual skills',
        'job skills',
        'key skills',
        'new skills',
        'or skill',
        'other skills',
        'personal skills',
        'preferred skills',
        'primary skills',
        'professional skills',
        'related skills',
        'required job skills',
        'required skills',
        'skill',
        'skills',
        'special skills',
        'specific skills',
        'welcomed skill',
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(
            endings=['skills', 'skill'],
            stop_phrases=self.GENERIC_SKILL_PHRASES,
            *args,
            **kwargs
        )


class AbilityPatternExtractor(NPEndPatternExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            endings=['ability', 'abilities'],
            stop_phrases=[],
            *args,
            **kwargs
        )
