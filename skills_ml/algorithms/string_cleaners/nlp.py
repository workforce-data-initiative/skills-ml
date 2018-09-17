"""String transformations for cleaning
for unicodedata, see:
http://www.unicode.org/reports/tr44/tr44-4.html#General_Category_Values
"""
import unicodedata
import re
from bs4 import BeautifulSoup
import nltk
from functools import reduce, wraps
from typing import List, Set, Generator, Dict, Pattern

transforms = ['nlp_a']
BULLET_CHARACTERS = ['+', '*', '-']


def deep(func):
    """A decorator that will apply a function to a nested list recursively

    Args:
        func (function): a function to be applied to a nested list
    Returns:
        function: The wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], list):
            if len(args[0]) == 1:
                return wrapper(args[0][0])
            return [wrapper(i) for i in args[0]]
        return func(args[0])
    return wrapper


def normalize(text: str) -> str:
    """
    Args:
        text (str): A unicode string
    Returns:
        str: The text, lowercased and in NFKD normal form
    """
    return unicodedata.normalize('NFKD', text.lower())

@deep
def clean_html(text: str) -> str:
    markup = BeautifulSoup(text, "lxml")
    return unicodedata.normalize('NFKD', markup.get_text())


def lowercase_strip_punc(text: str, punct: Set[str]=None) -> str:
    """
    Args:
        text (str): A unicode string
        punct (:obj: `set`, optional)
    Returns:
        str: The text, lowercased, sans  punctuation and in NFKD normal form
    """
    if not punct:
        punct = set(['P', 'S'])

    return ''.join(
        char for char in normalize(text)
        if not unicodedata.category(char)[0] in punct
    )


def title_phase_one(text: str, punct: Set[str]=None) -> str:
    """
    Args:
        text (str): A unicode string
        punct (:obj: `set`, optional)
    Returns:
        str: The text, lowercased, sans punctuation, whitespace normalized
    """
    if not punct:
        punct = set(['P', 'S'])
    no_apos = re.sub(r'\'', '', normalize(text))
    strip_punc = ''.join(
        char if not unicodedata.category(char)[0] in punct else ' '
        for char in no_apos
    )
    return re.sub(r'\s+', ' ', strip_punc.strip())

@deep
def clean_str(text: str) -> str:
    """
    Args:
        text: A unicode string
    Returns:
        str: lowercased, sans punctuation, non-English letters
    """
    RE_PREPROCESS = r'\W+|\d+'
    text = re.sub(
        RE_PREPROCESS,
        ' ',
        text.lower()
    )
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def sentence_tokenize(text: str) -> List[str]:
    """
    Args:
        text (str): a unicode string
    Returns:
        list: tokenized sentence
    """
    sentences = re.split('\n', text)
    sentences = list(filter(None, sentences))
    try:
        sentences = reduce(lambda x, y: x+y, map(lambda x: nltk.sent_tokenize(x), sentences.encode('utf-8')))
    except:
        sentences = reduce(lambda x, y: x+y, map(lambda x: nltk.sent_tokenize(x), sentences))
    return sentences

@deep
def word_tokenize(text: str, punctuation=True) -> List[str]:
    """
    Args:
        text (str): a unicode string
    Returns:
        list: tokenized words
    """
    if punctuation:
        return nltk.word_tokenize(text)
    else:
        return nltk.wordpunct_tokenize(text)


def fields_join(
        document: Dict,
        document_schema_fields: List[str]=None) -> str:
    """
    Args:
        document (dict): a document dictionary
        document_schema_fields (:obj: `list`, optional): a list of keys
    Returns:
        str: a text joined with selected fields.
    """
    if not document_schema_fields:
        document_schema_fields = [
                'description',
                'experienceRequirements',
                'qualifications',
                'skills']
    return ' '.join([document.get(field, '') for field in document_schema_fields])


def vectorize(
        tokenized_text: List[str],
        embedding_model):
    """
    Args:
        tokenized_text: a tokenized list of word tokens
        embedding_model: the embedding model implements `.infer_vector()` method
    Returns:
        np.ndarray: a word embedding vector
    """
    return embedding_model.infer_vector(tokenized_text)


def section_extract(section_regex: Pattern, document: str) -> List:
    """Only return the contents of the configured section heading

    Defines a 'heading' as the text of a sentence that:
        - does not itself start with a bullet character
        - either has between 1 and 3 words or ends in a colon

    For a heading that matches the given pattern, returns each sentence between it and the next heading.

    Heavily relies on the fact that sentence_tokenize does line splitting
    as well as standard sentence tokenization. In this way, it should work both
    for text strings that have newlines and for text strings that don't.

    In addition, this function splits each sentence by bullet characters as often bullets denote
    what we want to call 'sentences', but authors often take advantage of the bullet characters
    to make the contents of each 'sentence' into small sentence fragments, which makes standard
    sentence tokenization insufficient if the newlines have been taken out.
    """
    units_in_section = []
    sentences = sentence_tokenize(document)
    units = [
        unit
        for sentence in sentences
        for unit in split_by_bullets(sentence)
    ]

    heading = ''
    for unit in units:
        words_in_unit = len(unit.lstrip().rstrip().split(' '))
        if unit.strip() and unit[0] not in BULLET_CHARACTERS and ((words_in_unit > 0 and words_in_unit < 4) or unit.endswith(':')):
            heading = unit
        if re.match(section_regex, heading) and unit != heading and len(unit.strip()) > 0:
            units_in_section.append(unit.lstrip().rstrip())
    return units_in_section


def split_by_bullets(sentence: str) -> List:
    """Split sentence by bullet characters"""
    units = []
    for bullet_char in BULLET_CHARACTERS:
        padded_bullet = bullet_char + ' '
        if sentence.count(padded_bullet) > 1:
            for i, fragment in enumerate(sentence.split(padded_bullet)):
                if i > 0:
                    units.append(padded_bullet + fragment)
                else:
                    units.append(fragment)
            return units
    units.append(sentence)
    return units


def strip_bullets_from_line(line: str) -> str:
    """Remove bullets from beginning of line"""
    for bullet_char in BULLET_CHARACTERS:
        if line.startswith(bullet_char):
            line = line.replace(bullet_char, '')
    return line
