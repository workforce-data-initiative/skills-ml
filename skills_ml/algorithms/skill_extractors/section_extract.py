from .base import SkillExtractor, CandidateSkill, CandidateSkillYielder
from skills_ml.algorithms.nlp import section_extract
import logging
from typing import Dict


class SectionExtractSkillExtractor(SkillExtractor):
    """Extract skills from text by extracting sentences from matching 'sections'.

    Heavily utilizes skills_ml.algorithms.nlp.section_extract.
    For more detail on how to define 'sections', refer to its docstring.
    """
    def __init__(self, section_regex=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.section_regex = section_regex or r'.*([Qq]ualifications|[Ss]kills|[Rr]equirements|[Ab]ilities|[Cc]ompetencies)'

    @property
    def name(self):
        return f'section_extract_{self.section_regex}'

    @property
    def description(self):
        return f'Sentences from section matching regular expression: {self.section_regex}'

    def candidate_skills(self, source_object: Dict) -> CandidateSkillYielder:
        """Generate candidate skills from the source object

        Yields each sentence from the configured section pattern
        """

        lines_in_section = section_extract(self.section_regex, source_object['description'])
        for line in lines_in_section:
            logging.info('Yielding candidate skill %s', line)
            yield CandidateSkill(
                skill_name=line,
                matched_skill_identifier=None,
                confidence=100,
                context=line,
                start_index=-1,
                document_id=source_object['id'],
                document_type=source_object['@type'],
                source_object=source_object,
                skill_extractor_name=self.name
            )
