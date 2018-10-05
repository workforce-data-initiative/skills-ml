# coding: utf-8
"""Use noun phrases with specific endings to extract skills from job postings"""

import logging

import nltk
try:
    nltk.pos_tag('test')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    from nltk.tokenize.moses import MosesDetokenizer
except LookupError:
    nltk.download('perluniprops')
    from nltk.tokenize.moses import MosesDetokenizer

from .base import SkillExtractor, CandidateSkill, CandidateSkillYielder

from typing import Dict


def sentences_words_pos(document):
    """Chops raw text into part-of-speech (POS)-tagged words in sentences

    Args:
        document (string) A document in text format

    Returns: (list) of sentences, each being a list of word/POS pair

    Example:
        sentences_words_pos(
            '* Develop and maintain relationship with key members of ' +
            'ESPN\u2019s Spanish speaking editorial team'
        )
        [ # list of sentences
            [ # list of word/POS pairs
                ('*', 'NN'),
                ('Develop', 'NNP'),
                ('and', 'CC'),
                ('maintain', 'VB'),
                ('relationship', 'NN'),
                ('with', 'IN'),
                ('key', 'JJ'),
                ('members', 'NNS'),
                ('of', 'IN'),
                ('ESPN', 'NNP'),
                ('’', 'NNP'),
                ('s', 'VBD'),
                ('Spanish', 'JJ'),
                ('speaking', 'NN'),
                ('editorial', 'NN'),
                ('team', 'NN')
            ]
        ]
    """
    try:
        sentences = nltk.sent_tokenize(document.encode('utf-8'))
    except:
        sentences = nltk.sent_tokenize(document)
    sentences = [nltk.wordpunct_tokenize(sent) for sent in sentences]

    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


def noun_phrases_in_line_with_context(line):
    """Generate noun phrases in the given line of text

    Args:
        text (string): A line of raw text

    Yields:
        tuples, each with two strings:
            - a noun phrase
            - the context of the noun phrase (currently defined as the surrounding sentence)
    """
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
                        logging.debug('Yielding noun phrase %s with context %s', np, sent)
                        yield np, sent


BULLETS = ['+', '*', '-']


def is_bulleted(string):
    """Whether or not a given string begins a 'bullet' character

    A bullet character is understood to indicate list membership.
    Differeing common bullet characters are checked.

    Args:
    string (string): Any string

    Returns: (bool) whether or not the string begins with one of the characters
        in a predefined list of common bullets
    """
    string_first = string.split(' ', 1)[0].replace('\n', '')
    try:
        if string_first[0] in BULLETS:
            return True
        else:
            return False
    except:
        return False


BEGINNING_SUBSTRINGS_TO_REMOVE = [
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


def clean_beginning(string):
    """Clean the beginning of a string of common undesired formatting substrings

    Args:
    string (string): Any string

    Returns: The string with beginning formatting substrings removed
    """
    for substring in BEGINNING_SUBSTRINGS_TO_REMOVE:
        if string.startswith(substring):
            return string.replace(substring, "")
    return string


class NPEndPatternExtractor(SkillExtractor):
    """Identify noun phrases with certain ending words (e.g 'skills', 'abilities') as skills

    Args:
        endings (list): Single words that should identify the ending of a noun phrase
            as being a skill
        stop_phrases (list): Noun phrases that should not be considered skills
        only_bulleted_lines (bool, default True): Whether or not to only consider lines
            that look like they are items in a list
    """
    def __init__(self, endings, stop_phrases, only_bulleted_lines=True, confidence=95,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endings = endings
        self.stop_phrases = stop_phrases
        self.only_bulleted_lines = only_bulleted_lines
        self.detokenizer = MosesDetokenizer()
        self.confidence = confidence

    @property
    def description(self):
        return f'Noun phrases ending with one of: {self.endings}\n'
        f'Excluding phrases: {self.stop_phrases}\n'
        f'Only using bulleted lines: {self.only_bulleted_lines}'

    def candidate_skills(self, source_object: Dict) -> CandidateSkillYielder:
        """Generate candidate skills from the source object

        Looks for all noun phrases with one of the configured endings.

        Args:
            source_object (dict) A single source object to extract skills from.

        Yields: all candidate skills (algorithms.skill_extractors.base.CandidateSkill)
            found in the job posting
        """
        document = self.transform_func(source_object)
        for cleaned_phrase, context, phrase_start in self.noun_phrases_matching_endings(document):
            orig_context = self.detokenizer.detokenize([t[0] for t in context], return_str=True)
            logging.info(
                'Yielding candidate skill %s in context %s',
                cleaned_phrase,
                orig_context
            )
            yield CandidateSkill(
                skill_name=cleaned_phrase,
                matched_skill_identifier=None,
                confidence=self.confidence,
                context=orig_context,
                start_index=phrase_start,
                document_id=source_object['id'],
                document_type=source_object['@type'],
                source_object=source_object,
                skill_extractor_name=self.name
            )

    def noun_phrases_matching_endings(self, document):
        """From the given document, generate noun phrases ending with one of the configured terms

        Args:
            document (string) A raw text document, such as a job posting

        Yields:
            tuples, each with:
                - a noun phrase
                - the context of the noun phrase (currently defined as the surrounding sentence)
                - the index of the start of the phrase in the document
        """
        lines = document.split('\n')
        phrase_start = 0
        for line in lines:
            if not self.only_bulleted_lines or is_bulleted(line):
                for noun_phrase, context in noun_phrases_in_line_with_context(line):
                    term_list = noun_phrase.split()
                    if term_list[-1].lower() in self.endings:
                        cleaned_phrase = clean_beginning(noun_phrase).lower()
                        if cleaned_phrase not in self.stop_phrases:
                            yield cleaned_phrase, context, phrase_start
                    phrase_start += len(noun_phrase)


class SkillEndingPatternExtractor(NPEndPatternExtractor):
    """Identify noun phrases ending with 'skill' or 'skills' as skills"""

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
        'essential job skills',
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

    @property
    def name(self):
        bulleted_str = '_bulleted' if self.only_bulleted_lines else ''
        return f'noun_phrase_skill{bulleted_str}'


class AbilityEndingPatternExtractor(NPEndPatternExtractor):
    """Identify noun phrases ending in 'ability' or 'abilities' as skills"""
    def __init__(self, *args, **kwargs):
        super().__init__(
            endings=['ability', 'abilities'],
            stop_phrases=['demonstrated ability', 'ability'],
            *args,
            **kwargs
        )

    @property
    def name(self):
        bulleted_str = '_bulleted' if self.only_bulleted_lines else ''
        return f'noun_phrase_ability{bulleted_str}'
