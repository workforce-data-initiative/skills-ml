import csv
import pandas as pd
import logging


class OnetSkillImportanceExtractor(object):
    """
    An object that creates a skills importance CSV based on ONET data
    """
    def __init__(self, onet_source, output_filename, hash_function):
        """
        Args:
            output_filename: A filename to write the final dataset
            onet_source: An object that is able to fetch ONET files by name
            hash_function: A function that can hash a given string
        """
        self.output_filename = output_filename
        self.onet_source = onet_source
        self.hash_function = hash_function

    def onet_to_pandas(self, filename, col_name):
        """
        Args:
            filename: an unpathed filename referring to an ONET skill file
            col_name: A list of columns to extract from the file

        Returns:
            A pandas DataFrame
        """
        logging.info('Converting ONET %s to pandas', filename)
        with self.onet_source.ensure_file(filename) as fullpath:
            with open(fullpath) as f:
                onet = [row for row in csv.DictReader(f, delimiter='\t')]
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
            'Scale ID',
            'Data Value',
            'N',
            'Standard Error',
            'Lower CI Bound',
            'Upper CI Bound',
            'Recommend Suppress',
            'Not Relevant',
            'Date',
            'Domain Source'
        ]
        skills = self.onet_to_pandas('Skills.txt', standard_columns)
        ability = self.onet_to_pandas('Abilities.txt', standard_columns)
        knowledge = self.onet_to_pandas('Knowledge.txt', standard_columns)

        # Concat KSA dataframes into one table
        standard_columns[2] = 'ONET KSA'
        skills.columns = standard_columns
        ability.columns = standard_columns
        knowledge.columns = standard_columns

        onet_ksas = pd.concat(
            (skills, ability, knowledge),
            ignore_index=True
        )

        onet_ksas['skill_uuid'] = onet_ksas['ONET KSA']\
            .apply(self.hash_function)

        onet_ksas.to_csv(self.output_filename, sep='\t')
