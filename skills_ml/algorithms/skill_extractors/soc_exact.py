from skills_ml.ontologies.base import CompetencyOntology

from .exact_match import ExactMatchSkillExtractor


class SocScopedExactMatchSkillExtractor(ExactMatchSkillExtractor):
    """Extract skills from unstructured text,
    but only return matches that agree with a known taxonomy
    """
    method_name = 'occscoped_exact_match'
    method_description = 'Exact matching using only the skills known by ONET to be associated with the given SOC code'

    def __init__(self, competency_ontology, *args, **kwargs):
        if not isinstance(competency_ontology, CompetencyOntology):
            raise ValueError('Must pass in a CompetencyOntology object')
        super().__init__(competency_ontology.competency_framework, *args, **kwargs)

        self.competency_ontology = competency_ontology
        self.skill_extractor_by_soc = {}

    def candidate_skills(self, source_object):
        soc_code = source_object.get('onet_soc_code', None)
        if not soc_code:
            return
        if soc_code not in self.skill_extractor_by_soc:
            subontology = self.competency_ontology.filter_by(
                lambda edge: edge.occupation.identifier == soc_code,
                competency_name='competencies_' + soc_code,
                competency_description='Competencies needed by ' + soc_code
            )
            if len(subontology.competencies) == 0:
                return

            self.skill_extractor_by_soc[soc_code] = ExactMatchSkillExtractor(subontology.competency_framework)
        yield from self.skill_extractor_by_soc[soc_code].candidate_skills(source_object)
