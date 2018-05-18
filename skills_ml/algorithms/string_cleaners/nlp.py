"""String transformations for cleaning"""
import unicodedata
import re
from bs4 import BeautifulSoup
import nltk
from functools import reduce

class NLPTransforms(object):
    # An object that performs common NLP transformations
    # for unicodedata, see:
    # http://www.unicode.org/reports/tr44/tr44-4.html#General_Category_Values
    def __init__(self):
        self.punct = set(['P', 'S'])
        self.transforms = ['nlp_a']

    def normalize(self, text):
        """
        Args:
            text: A unicode string
        Returns:
            The text, lowercased and in NFKD normal form
        """
        return unicodedata.normalize('NFKD', text.lower())

    def clean_html(self, text):
        markup = BeautifulSoup(text, "lxml")
        return unicodedata.normalize('NFKD', markup.get_text())

    def lowercase_strip_punc(self, text):
        """
        Args:
            text: A unicode string
        Returns:
            The text, lowercased, sans  punctuation and in NFKD normal form
        """
        return ''.join(
            char for char in self.normalize(text)
            if not unicodedata.category(char)[0] in self.punct
        )

    def title_phase_one(self, text):
        """
        Args:
            text: A unicode string
        Returns:
            The text, lowercased, sans punctuation, whitespace normalized
        """
        no_apos = re.sub(r'\'', '', self.normalize(text))
        strip_punc = ''.join(
            char if not unicodedata.category(char)[0] in self.punct else ' '
            for char in no_apos
        )
        return re.sub(r'\s+', ' ', strip_punc.strip())

    def clean_str(self, text):
        """
        Args:
            text: A unicode string
        Returns:
            The array of split words in text, lowercased,
            sans punctuation, non-English letters
        """
        RE_PREPROCESS = r'\W+|\d+'
        text = re.sub(
            RE_PREPROCESS,
            ' ',
            self.lowercase_strip_punc(text)
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

    def sentence_tokenize(self, text):
        sentences = re.split('\n', text)
        sentences = list(filter(None, sentences))
        try:
            sentences = reduce(lambda x, y: x+y, map(lambda x: nltk.sent_tokenize(x), sentences.encode('utf-8')))
        except:
            sentences = reduce(lambda x, y: x+y, map(lambda x: nltk.sent_tokenize(x), sentences))
        return sentences

    def word_tokenize(self, text, punctuation=True):
        if punctuation:
            return nltk.word_tokenize(text)
        else:
            return nltk.wordpunct_tokenize(text)
