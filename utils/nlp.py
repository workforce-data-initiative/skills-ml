"""
Shared Natural Language Processing utilities
"""
import unicodedata


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
