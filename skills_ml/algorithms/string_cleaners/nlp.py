"""String transformations for cleaning
for unicodedata, see:
http://www.unicode.org/reports/tr44/tr44-4.html#General_Category_Values
"""
import unicodedata
import re
from bs4 import BeautifulSoup
import nltk
from functools import reduce, wraps
from typing import List, Set, Generator, Dict


transforms = ['nlp_a']

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
