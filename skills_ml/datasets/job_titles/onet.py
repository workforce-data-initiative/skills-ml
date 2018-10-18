"""Process ONET job titles into a common format"""
import pandas as pd

from skills_ml.algorithms.nlp import transforms, lowercase_strip_punc


class Onet_Title(object):
    """An object representing job title data from different ONET files

    Originally written by Kwame Porter Robinson
    """
    def __init__(self, onet_cache):
        """
            Args:
                onet_cache: an object that is able to fetch ONET files by name
        """
        self.onet_cache = onet_cache

        self.occupation = {}
        self.occupation['name'] = 'Occupation Data.txt'
        self.occupation['fields'] = ['O*NET-SOC Code', 'Title']

        self.alternative = {}
        self.alternative['name'] = 'Alternate Titles.txt'
        self.alternative['fields'] = ['O*NET-SOC Code', 'Alternate Title']

        self.sample = {}
        self.sample['name'] = 'Sample of Reported Titles.txt'
        self.sample['fields'] = ['O*NET-SOC Code', 'Reported Job Title']

    def extract(self, name, columns):
        """
            Args:
                name: unpathed filename of an ONET file ('Occupation Data.txt')
                columns: a list of columns to extract from the file
        """
        with self.onet_cache.ensure_file(name) as full_path:
            return pd.read_csv(full_path, delimiter='\t')[columns]


class OnetTitleExtractor(object):
    """
        An object that creates a job titles CSV based on ONET data
    """
    def __init__(self, output_filename, onet_source, hash_function):
        """
            Args:
                output_filename: A filename to write the final dataset
                onet_source: An object that is able to fetch ONET files by name
                hash_function: A function that can hash a given string
        """
        self.output_filename = output_filename
        self.onet_source = onet_source
        self.hash_function = hash_function

    def run(self):
        """
            Creates a job titles CSV based on ONET occupation and title data
        """
        titles = Onet_Title(self.onet_source)
        # TODO: Get descriptions, original title
        onet_titles = titles.extract(titles.occupation['name'],
                                     titles.occupation['fields'])
        alternative_titles = titles.extract(titles.alternative['name'],
                                            titles.alternative['fields'])
        sample_titles = titles.extract(titles.sample['name'],
                                       titles.sample['fields'])

        alternative_titles.columns = onet_titles.columns
        sample_titles.columns = onet_titles.columns

        job_titles = pd.concat(
            (onet_titles, alternative_titles, sample_titles)
        )
        unique_titles = titles.extract(
            titles.occupation['name'],
            titles.occupation['fields'] + ['Description']
        ).drop_duplicates()
        unique_titles.columns = [
            'O*NET-SOC Code',
            'Original Title',
            'Description'
        ]

        unique_titles['job_uuid'] = unique_titles['Original Title']\
            .apply(self.hash_function)
        titles_complete = pd.merge(job_titles,
                                   unique_titles,
                                   how='left',
                                   on=['O*NET-SOC Code'])
        titles_complete[transforms[0]] = titles_complete['Title']\
            .apply(lowercase_strip_punc)
        titles_complete.to_csv(self.output_filename, sep='\t')
