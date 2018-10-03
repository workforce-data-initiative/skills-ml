from copy import deepcopy
import json
import logging
import unicodecsv as csv
from statistics import mean

import s3fs
from descriptors import cachedproperty

from skills_utils.iteration import Batch
from skills_utils.s3 import S3BackedJsonDict
from skills_utils.hash import md5
from skills_ml.algorithms.skill_extractors.base import CandidateSkill
from skills_ml.algorithms.sampling import Sample
from skills_ml.storage import store_from_path
from skills_ml.job_postings import JobPosting

from typing import Dict

AnnotationType = Dict


class BratExperiment(object):
    """Manage a BRAT experiment. Handles:

    1. The creation of BRAT config for a specific sample of job postings
    2. Adding users to the installation and allocating them semi-hidden job postings
    3. The parsing of the annotation results at the end of the experiment

    Syncs data to an experiment directory on S3.
    BRAT installations are expected to sync this data down regularly.

    Keeps track of a metadata file,
    available as a dictionary at self.metadata, with the following structure:

    # these first five keys are just storage of user input to either
    # the constructor or start()
    # view relevant docstrings for definitions
    sample_base_path
    sample_name
    entities_with_shortcuts
    minimum_annotations_per_posting
    max_postings_per_allocation

    # units and allocations are far more important when reading results of an experiment
    units: {
        # canonical list of 'unit' (bundle of job postings) names,
        # along with a list of tuples of job posting keys (only unique within unit)
            and globally unique job posting ids
        'unit_1': [
            (posting_key_1, job_posting_id_1),
            (posting_key_2, job_posting_id_2),
        ],
        'unit_2': [
            (posting_key_1, job_posting_id_3),
            (posting_key_2, job_posting_id_4),
        ]
    }
    allocations: {
        # canonical list of unit assignments to users
        'user_1': ['unit_1', 'unit_2'],
        'user_2': ['unit_2']
    }
    """
    def __init__(self, experiment_name, brat_s3_path):
        self.experiment_name = experiment_name
        self.brat_s3_path = brat_s3_path
        self.metadata = S3BackedJsonDict(
            path=self.experiment_path + '/metadata'
        )
        self.user_pw_store = S3BackedJsonDict(
            path=self.experiment_path + '/user-pws'
        )
        self.s3 = s3fs.S3FileSystem()

    @property
    def experiment_path(self):
        "The s3 path to all files relating to the experiment"
        return '/'.join([self.brat_s3_path, self.experiment_name])

    @property
    def brat_config_path(self):
        "The s3 path to BRAT config files for the experiment"
        return '/'.join([self.experiment_path, 'brat_config'])

    @property
    def data_path(self):
        "The s3 path to job postings for the experiment"
        return '/'.join([self.brat_config_path, 'data'])

    def unit_path(self, unit_name):
        "The s3 path to job postings for a particular unit"
        return '/'.join([self.data_path, '.' + unit_name])

    def user_allocations_path(self, user_name):
        "The s3 path to all allocations for a user (i.e what they should see when logging in"
        return '/'.join([self.data_path, '.' + user_name])

    def allocation_path(self, user_name, unit_name):
        "The s3 path for a particular allocation for a user"
        return '/'.join([self.user_allocations_path(user_name), unit_name])

    def start(
        self,
        sample,
        entities_with_shortcuts,
        minimum_annotations_per_posting=2,
        max_postings_per_allocation=10,
    ):
        """Starts a BRAT experiment by dividing up a job posting sample into units
            and creating BRAT config files

        Args:
            sample (skills_ml.algorithms.sampling.Sample) A sample of job postings
            entities_with_shortcuts (collection of tuples) The distinct entities to tag.
                The first entry of each tuple should be a one character string
                    that can be used as a keyboard shortcut in BRAT
                The second entry of each tuple should be the name of the entity
                    that shows up in menus
            minimum_annotations_per_posting (int, optional) How many people should annotate
                each job posting before allocating new ones. Defaults to 2
            max_postings_per_allocation (int, optional) How many job postings for each allocation.
                Should be a number that is not so high as to be daunting for users at first,
                but not so low as to make it a hassle to do several given that requesting
                new allocations is not automatic.
                Defaults to 10
        """
        logging.info('Starting experiment! Wait for a bit')
        self.metadata['sample_base_path'] = sample.base_path
        self.metadata['sample_name'] = sample.name
        self.metadata['entities_with_shortcuts'] = entities_with_shortcuts
        self.metadata['minimum_annotations_per_posting'] = minimum_annotations_per_posting
        self.metadata['max_postings_per_allocation'] = max_postings_per_allocation

        # 1. Output job posting text
        self.metadata['units'] = {}
        logging.info('Dividing sample into bundles of size %s', max_postings_per_allocation)
        for unit_num, batch_postings in enumerate(Batch(sample, max_postings_per_allocation)):
            unit_name = 'unit_{}'.format(unit_num)
            self.metadata['units'][unit_name] = []
            for posting_key, posting_string in enumerate(batch_postings):
                posting = JobPosting(posting_string)
                self.metadata['units'][unit_name].append((posting_key, posting.id))
                outfilename = '/'.join([self.unit_path(unit_name), str(posting_key)])
                logging.info('Writing to %s', outfilename)
                with self.s3.open(outfilename + '.txt', 'wb') as f:
                    f.write(posting.text.encode('utf-8'))
                with self.s3.open(outfilename + '.ann', 'wb') as f:
                    f.write(''.encode('utf-8'))
        self.metadata.save()
        logging.info('Done creating bundles. Now creating BRAT configuration')

        # 2. Output annotation.conf with lists of entities
        with self.s3.open('/'.join([self.brat_config_path, 'annotation.conf']), 'wb') as f:
            f.write('[entities]\n'.encode('utf-8'))
            for _, entity_name in entities_with_shortcuts:
                f.write(entity_name.encode('utf-8'))
                f.write('\n'.encode('utf-8'))
            f.write('[relations]\n\n# none defined'.encode('utf-8'))
            f.write('[attributes]\n\n# none defined'.encode('utf-8'))
            f.write('[events]\n\n# none defined'.encode('utf-8'))

        # 3. Output kb_shortcuts.conf with quick type selection for each entity
        with self.s3.open('/'.join([self.brat_config_path, 'kb_shortcuts.conf']), 'wb') as f:
            for shortcut, entity_name in entities_with_shortcuts:
                to_write = shortcut + ' ' + entity_name + '\n'
                f.write(to_write.encode('utf-8'))

        # 4. Output visual.conf with list of entities
        with self.s3.open('/'.join([self.brat_config_path, 'visual.conf']), 'wb') as f:
            f.write('[labels]\n'.encode('utf-8'))
            for _, entity_name in entities_with_shortcuts:
                f.write(entity_name.encode('utf-8'))
                f.write('\n'.encode('utf-8'))

        logging.info('Done creating BRAT configuration. All data is at %s', self.experiment_path)

    def add_user(self, username, password):
        """Creates a user with an allocation

        Args:
            username (string) The desired username
            password (string) The desired password
        """
        # Creates a user with an allocation.

        if username in self.user_pw_store:
            raise ValueError('User {} already created'.format(username))
        self.user_pw_store[username] = password
        self.user_pw_store.save()
        return self.add_allocation(username)

    def needs_allocation(self, unit_name):
        """Whether or not this unit needs to be allocated again.

        Args:
            unit_name (string) The name of a unit (from experiment's .metadata['units']

        Returns: (bool) Whether or not the unit should be allocated again
        """
        return sum([
            1
            for user_units in self.metadata['allocations'].values()
            if unit_name in user_units
        ]) < self.metadata['minimum_annotations_per_posting']

    def add_allocation(self, user_name):
        """Allocate a unit of job postings to the given user

        Args:
            user_name (string) A username (expected to be created already with a password)

        Returns: (string) The directory containing job postings in the allocation
        """
        # given a user name
        if user_name not in self.user_pw_store:
            raise ValueError('Username not in user-password store. Please call add_user first')

        # initialize allocations if there have been none yet
        if 'allocations' not in self.metadata:
            self.metadata['allocations'] = {}
        if user_name not in self.metadata['allocations']:
            self.metadata['allocations'][user_name] = []

        # see if there is a next unit that the user hasn't seen and really needs allocation
        unit_to_allocate = None
        try:
            unit_to_allocate = next(
                unit_name
                for unit_name in self.metadata['units'].keys()
                if unit_name not in self.metadata['allocations'][user_name]
                and self.needs_allocation(unit_name)
            )
        except StopIteration:
            pass

        # if there is none that really needs allocation
        # just pick the next one they haven't seen yet
        if not unit_to_allocate:
            try:
                unit_to_allocate = next(
                    unit_name
                    for unit_name in self.metadata['units'].keys()
                    if unit_name not in self.metadata['allocations'][user_name]
                )
            except StopIteration:
                pass

        if not unit_to_allocate:
            raise ValueError('No units left to allocate to user!')

        # create and populate a directory for the user that has the contents of the unit
        source_dir = self.unit_path(unit_to_allocate)
        dest_dir = self.allocation_path(user_name, unit_to_allocate)

        for source_key in self.s3.ls(source_dir):
            dest_key = source_key.replace(source_dir, dest_dir)
            self.s3.copy(source_key, dest_key)

        # record in metadata the fact that the user has been allocated this
        self.metadata['allocations'][user_name].append(unit_to_allocate)
        self.metadata.save()
        logging.info('Allocation created! Directory is %s', dest_dir)
        return dest_dir

    @cachedproperty
    def annotations_by_unit(self):
        """Fetch raw annotations by unit

        Structure of return dictionary is as follows.
            {'unit_name': {'posting_key': {'user_name': [annotations...]}}}
            each annotation is a dictionary of: {
                'entity' # the name of the configured entity (e.g. Skill) that the user annotated
                'start_index': # the start index of the annotation in the text
                'end_index': # the end index of the annotation in the text
                'labeled_string': # the text string that the user annotated,
            }
        Returns: (dict) The annotations by unit, posting key, and user name
        """
        # create annotations by unit
        annotations_by_unit = {}
        for user_name, unit_names in self.metadata['allocations'].items():
            for unit_name in unit_names:
                if unit_name not in annotations_by_unit:
                    annotations_by_unit[unit_name] = {}
                allocation_path = self.allocation_path(user_name, unit_name)
                for key in self.s3.ls(allocation_path + '/'):
                    # this will iterate through both posting text (.txt) and annotation (.ann)
                    if key.endswith('.ann'):
                        posting_key = key.split('/')[-1].replace('.ann', '')
                        with self.s3.open(key) as f:
                            logging.info('Reading annotation file at %s', key)
                            raw_annotations = csv.reader(f, delimiter='\t')
                            if posting_key not in annotations_by_unit[unit_name]:
                                annotations_by_unit[unit_name][posting_key] = {}

                            converted_annotations = []
                            for annotation in raw_annotations:
                                logging.info('Found annotation line %s', annotation)
                                index, middle, string = annotation
                                entity, start, end = middle.split(' ')
                                converted_annotations.append({
                                    'entity': entity,
                                    'start_index': int(start),
                                    'end_index': int(end),
                                    'labeled_string': string,
                                })
                        annotations_by_unit[unit_name][posting_key][user_name] = \
                            converted_annotations

        return annotations_by_unit

    @cachedproperty
    def sequence_tagged_annotations(self):
        """Fetch sequence tagged annotations

        Expects these annotations to be produced by BRAT in CoNLL format.
        Returns: (dict), keys are tuples of (job posting id, tagger_id)
            and values are lists of (entity, token) tuples
        """
        annotations_by_posting_and_user = {}
        for user_name, unit_names in self.metadata['allocations'].items():
            for unit_name in unit_names:
                posting_id_lookup = dict(self.metadata['units'][unit_name])
                allocation_path = self.allocation_path(user_name, unit_name)
                for key in self.s3.ls(allocation_path + '/'):
                    # this will iterate through posting text (.txt), annotation (.ann),
                    # and CoNLL (.conll) files. In this case we only care about conll
                    if key.endswith('.conll'):
                        posting_key = key.split('/')[-1].replace('.conll', '')
                        with self.s3.open(key) as f:
                            logging.info('Reading conll file at %s', key)
                            job_posting_id = posting_id_lookup[int(posting_key)]
                            raw_tokens = csv.reader(f, delimiter='\t')

                            tokens = []
                            for token_line in raw_tokens:
                                logging.info('Found token line %s', token_line)
                                if len(token_line) == 0:
                                    tokens.append((None, None))
                                else:
                                    tag, _, _, token = token_line
                                    tokens.append((tag, token))

                            key = (job_posting_id, md5(user_name))
                            if any(token for token in tokens if token[0] not in {'O', None}):
                                annotations_by_posting_and_user[key] = tokens
                            else:
                                logging.warning('No annotations found in file. Skipping')

        return annotations_by_posting_and_user

    def average_observed_agreement(self):
        """Calculate average observed agreement by unit and posting key

        Computed *per-label* and averaged together.

        As an example:
        Assume a job posting has four total annotations from two separate annotators.
        Of these four, three are distinct. Each annotator agreed on one annotation, and each
        of them had an extra one that didn't agree with each other.
        The calculation is the mean of: [
            1.0 (the one they agreed on)
            0.5 (the one that annotator one tagged on their own)
            0.5 (the one that annotator two tagged on their own)
        ]
        The end result for this job posting will be 1.0+0.5+0.5/3, or ~0.667

        Structure of return dictionary is as follows.
            {'unit_name': {'posting_key': 0.5}}
        Returns: (dict) The average observed agreement by unit and posting key
        """
        # calculate average observed agreement
        agreement = deepcopy(self.labels_with_agreement_by_unit)
        for unit_name, posting_annotations in self.labels_with_agreement_by_unit.items():
            for posting_key, annotations in posting_annotations.items():
                if len(annotations) > 0:
                    agreement[unit_name][posting_key] = mean(
                        annotation['percent_tagged']
                        for annotation in annotations
                    )
                else:
                    agreement[unit_name][posting_key] = None
        return agreement

    @cachedproperty
    def labels_with_agreement_by_unit(self):
        """Fetch annotations with agreement by unit and job posting

        Structure of return dictionary is as follows.
            {'unit_name': {'posting_key': [annotations...]}}
            each annotation is a dictionary of: {
                'entity' # the name of the configured entity (e.g. Skill) that the user/s annotated
                'start_index': # the start index of the annotation in the text
                'end_index': # the end index of the annotation in the text
                'labeled_string': # the text string that the user/s annotated
                'percent_tagged': # of users that saw this posting, what percentage tagged this
                'number_seen' # how many users saw this job posting
            }
        Returns: (dict) The annotations by unit and posting key
        """
        labels_with_agreement = deepcopy(self.annotations_by_unit)
        for unit_name, posting_annotations in self.annotations_by_unit.items():
            for posting_key, user_annotations in posting_annotations.items():
                # to obtain a matrix, use the start/end indices as keys
                keyed_user_annotations = {}
                all_users = user_annotations.keys()
                for user_name, converted_annotations in user_annotations.items():
                    for converted_annotation in converted_annotations:
                        key = (
                            converted_annotation['start_index'],
                            converted_annotation['end_index'],
                            converted_annotation['entity'],
                            converted_annotation['labeled_string']
                        )
                        if key not in keyed_user_annotations:
                            keyed_user_annotations[key] = []
                        keyed_user_annotations[key].append(user_name)

                merged_annotations = []
                for key, users_who_tagged in keyed_user_annotations.items():
                    users_who_did_not_tag = [u for u in all_users]
                    for user in users_who_tagged:
                        users_who_did_not_tag.remove(user)
                    percentage_tagged = len(users_who_tagged) / len(all_users)
                    merged_annotations.append({
                        'start_index': key[0],
                        'end_index': key[1],
                        'entity': key[2],
                        'labeled_string': key[3],
                        'percent_tagged': percentage_tagged,
                        'number_seen': len(all_users)
                    })
                labels_with_agreement[unit_name][posting_key] = \
                    sorted(merged_annotations, key=lambda k: k['start_index'])

        return labels_with_agreement

    @cachedproperty
    def sample_lookup(self):
        if 'sample_base_path' not in self.metadata or 'sample_name' not in self.metadata:
            raise ValueError('Sample information needs to be available to look up sample. Have you run .start on this BratExperiment yet?')
        sample = Sample(store_from_path(self.metadata['sample_base_path']), self.metadata['sample_name'])
        lookup = {}
        for line in sample:
           obj = json.loads(line) 
           lookup[obj['id']] = obj
        return lookup

    def saved_text_lookup(self, unit, posting_key):
        textfilename = '/'.join([self.unit_path(unit), str(posting_key)])
        with self.s3.open(textfilename + '.txt', 'rb') as f:
            text = f.read().decode('utf-8')
            return text


    @cachedproperty
    def candidate_skills(self):
        """Format labels as CandidateSkills.

        Returns: (list) Flattened labels as CandidateSkill objects
        """
        by_unit = self.labels_with_agreement_by_unit
        candidate_skills = []
        for unit, unit_annotations in by_unit.items():
            posting_id_lookup = dict(self.metadata['units'][unit])
            for posting_key, posting_annotations in unit_annotations.items():
                logging.info(
                    'Looking up posting key %s in job posting id lookup %s',
                    posting_key,
                    posting_id_lookup
                )
                job_posting_id = posting_id_lookup[int(posting_key)]
                text = self.saved_text_lookup(unit, posting_key)
                for annotation in posting_annotations:
                    annotation['job_posting_id'] = job_posting_id
                    if annotation['start_index'] - 100 < 0:
                        context_start = 0
                    else:
                        context_start = annotation['start_index'] - 100
                    candidate_skills.append(CandidateSkill(
                        skill_name=annotation['labeled_string'],
                        matched_skill_identifier=None,
                        context=text[context_start:annotation['end_index']+100],
                        start_index=annotation['start_index'],
                        confidence=annotation['percent_tagged'],
                        document_id=job_posting_id,
                        document_type='JobPosting',
                        source_object=self.sample_lookup[job_posting_id],
                        skill_extractor_name='human_labeler',
                    ))
        return candidate_skills
