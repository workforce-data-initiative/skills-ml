"""
Shared Natural Language Processing utilities
"""
import unicodedata
import re


class NLPTransforms(object):
    # An object that performs common NLP transformations
    # for unicodedata, see:
    # http://www.unicode.org/reports/tr44/tr44-4.html#General_Category_Values
    def __init__(self):
        self.punct = set(['P', 'S'])
        self.transforms = ['nlp_a']

    def normalize(self, document):
        """
        Args:
            document: A unicode string
        Returns:
            The document, lowercased and in NFKD normal form
        """
        return unicodedata.normalize('NFKD', document.lower())

    def lowercase_strip_punc(self, document):
        """
        Args:
            document: A unicode string
        Returns:
            The document, lowercased, sans  punctuation and in NFKD normal form
        """
        return ''.join(
            char for char in self.normalize(document)
            if not unicodedata.category(char)[0] in self.punct
        )

    def title_phase_one(self, document):
        """
        Args:
            document: A unicode string
        Returns:
            The document, lowercased, sans punctuation, whitespace normalized
        """
        no_apos = re.sub(r'\'', '', self.normalize(document))
        strip_punc = ''.join(
            char if not unicodedata.category(char)[0] in self.punct else ' '
            for char in no_apos
        )
        return re.sub(r'\s+', ' ', strip_punc.strip())

    def clean_str(self, document):
        """
        Args:
            document: A unicode string
        Returns:
            The array of split words in document, lowercased,
            sans punctuation, non-English letters
        """
        RE_PREPROCESS = r'\W+|\d+'
        document = re.sub(
            RE_PREPROCESS,
            ' ',
            self.lowercase_strip_punc(document)
        )
        document = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", document)
        document = re.sub(r"\'s", " \'s", document)
        document = re.sub(r"\'ve", " \'ve", document)
        document = re.sub(r"n\'t", " n\'t", document)
        document = re.sub(r"\'re", " \'re", document)
        document = re.sub(r"\'d", " \'d", document)
        document = re.sub(r"\'ll", " \'ll", document)
        document = re.sub(r"\s{2,}", " ", document)
        return document
