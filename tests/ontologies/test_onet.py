from unittest import TestCase
from skills_ml.ontologies.onet import Onet
from skills_ml.ontologies.base import CompetencyOntology
from unittest.mock import patch

class OnetCompetencyTest(TestCase):
    @patch('skills_ml.ontologies.onet.Onet._build')
    def test_consturctor(self, _build_mock):
        onet = Onet()
        assert onet.is_built == False
        assert _build_mock.call_count == 1

    def test_onet_object(self):
        onet = Onet()
        assert isinstance(onet, CompetencyOntology)

        assert len(onet.all_soc) == 1110

        assert len(onet.all_major_groups) == 23

        assert onet.is_built == True

