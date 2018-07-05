"""Clean job titles by utilizing a list of stopwords
"""

import pandas as pd
import re
from collections import OrderedDict
import logging

from skills_ml.datasets.negative_positive_dict import negative_positive_dict

def clean_by_rules(jobtitle):
    """Remove numbers and normalize spaces

    Args:
        jobtitle (string) A string

    Returns: (string) the string with numbers removes and spaces normalized
    """
    # remove any words with number in it
    jobtitle = re.sub('\w*\d\w*', ' ', jobtitle).strip()

    # make one space between words
    jobtitle = ' '.join(jobtitle.split())

    return jobtitle

def clean_by_neg_dic(jobtitle, negative_list, positive_list):
    """Remove words from the negative dictionary

    Args:
        jobtitle (string) A job title string
        negative_list (collection) A list of stop words
        positive_list (collection) A list of positive words to override stop words

    Returns: (string) The cleaned job title
    """
    # Exact matching
    result = []
    for word in jobtitle.split():
        if word in negative_list:
            logging.debug('Found "%s" in negative dictionary', word)
        elif word in positive_list:
            logging.debug('Found "%s" in positive dictionary', word)
            result.append(word)
        else:
            result.append(word)
    result2str = ' '.join(result)

    return result2str

def aggregate(df_jobtitles, groupby_keys):
    """
    Args:
        df_jobtitles: job titles in pandas DataFrame
        groupby_keys: a list of keys to be grouped by. should be something like ['title', 'geo']
    Returns:
        agg_cleaned_jobtitles: a aggregated verison of job title in pandas DataFrame
    """
    agg_cleaned_jobtitles = pd.DataFrame(df_jobtitles.groupby(groupby_keys, as_index=False)['count'].sum())
    agg_cleaned_jobtitles = agg_cleaned_jobtitles.fillna('without jobtitle')

    return agg_cleaned_jobtitles

class JobTitleStringClean(object):
    """Clean job titles by stripping numbers, and removing place/state names (unless they are also ONET jobs)
    """

    def __init__(self):
        self.dict = negative_positive_dict()
        self.negative_list = self.dict['places'] + self.dict['states']
        self.positive_list = self.dict['onetjobs']

    def clean_title(self, title):
        return clean_by_neg_dic(
            clean_by_rules(title),
            self.negative_list,
            self.positive_list
        )

    def clean(self, df_jobtitles):
        """
        Clean the job titles by rules and negative dictionary.
        Args:
            df_jobtitles: job titles in pandas DataFrame
        Returns:
            cleaned_jobtitles: a clenaed verison of job title in pandas DataFrame
        """
        df_jobtitles = df_jobtitles.fillna('without jobtitle')

        columns = list(df_jobtitles.columns)
        cleaned_jobtitles = OrderedDict({key: [] for key in columns})
        progress_count = 0
        for i, row in enumerate(df_jobtitles.values):
            if progress_count % 1000 == 0:
                logging.info('%s/%s jobtitles have been cleaned.', progress_count, len(df_jobtitles))
            try:
                for colname in columns:
                    if colname == 'title':
                        new_title = clean_by_rules(row[columns.index(colname)])
                        new_title = clean_by_neg_dic(new_title, self.negative_list, self.positive_list)
                        cleaned_jobtitles[colname].append(new_title)
                    else:
                        cleaned_jobtitles[colname].append(row[columns.index(colname)])
                progress_count += 1
            except TypeError:
                logging.warning('There is a TypeError %s', row)

        cleaned_jobtitles = pd.DataFrame(cleaned_jobtitles)
        return cleaned_jobtitles
