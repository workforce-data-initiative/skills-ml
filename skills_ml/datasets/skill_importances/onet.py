"""Process ONET data to create a dataset with occupations and their skill importances"""
import csv
import pandas as pd
from skills_ml.datasets.onet_source import OnetToMemoryDownloader
import logging
import io
from skills_utils.hash import md5


class OnetSkillImportanceExtractor(object):
    """
    An object that creates a skills importance CSV based on ONET data

    Originally written by Kwame Porter Robinson
    """
    def __init__(self, storage, output_dataset_name, hash_function=None):
        """
        Args:
            output_dataset_name: A filename to write the final dataset
            onet_source: An object that is able to fetch ONET files by name
            hash_function: A function that can hash a given string
        """
        self.storage = storage
        self.output_dataset_name = output_dataset_name
        self.hash_function = hash_function or md5

    def onet_to_pandas(self, filename, col_name):
        """
        Args:
            filename: an unpathed filename referring to an ONET skill file
            col_name: A list of columns to extract from the file

        Returns:
            A pandas DataFrame
        """
        logging.info('Converting ONET %s to pandas', filename)
        filetext = OnetToMemoryDownloader().download(filename)
        onet = [row for row in csv.DictReader(io.StringIO(filetext), delimiter='\t')]
        onet = pd.DataFrame(onet)
        for col in col_name:
            onet[col] = onet[col].astype(str).str.lower()
        return onet[col_name]

    def run(self):
        """
            Creates a skills importance CSV based on ONET KSAT data,
        """
        # create dataframes for each KSA type
        standard_columns = [
            'O*NET-SOC Code',
            'Element ID',
            'Element Name',
        ]
        skills = self.onet_to_pandas('Skills', standard_columns)
        ability = self.onet_to_pandas('Abilities', standard_columns)
        knowledge = self.onet_to_pandas('Knowledge', standard_columns)
        tools = self.onet_to_pandas('Tools and Technology', ['O*NET-SOC Code', 'Commodity Code', 'T2 Example'])

        # Concat KSA dataframes into one table
        standard_columns[2] = 'ONET KSA'
        skills.columns = standard_columns
        ability.columns = standard_columns
        knowledge.columns = standard_columns
        tools.columns = standard_columns

        onet_ksas = pd.concat(
            (skills, ability, knowledge, tools),
            ignore_index=True
        )

        onet_ksas['nlp_a'] = onet_ksas['ONET KSA']\
            .apply(self.hash_function)

        fh = io.StringIO()
        onet_ksas.to_csv(fh, sep='\t')
        fh.seek(0)
        self.storage.write(fh.getvalue().encode('utf-8'), f'{self.output_dataset_name}.tsv')
