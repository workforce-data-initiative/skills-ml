from algorithms.skill_taggers.simple import SimpleSkillTagger
from tests import utils
from utils.hash import md5


def test_simple_skill_tagger():
    content = [
        ['', 'O*NET-SOC Code', 'Element ID', 'ONET KSA', 'Description', 'skill_uuid', 'nlp_a'],
        ['1', '11-1011.00', '2.a.1.a', 'reading comprehension', '...', '2c77c703bd66e104c78b1392c3203362', 'reading comprehension'],
        ['2', '11-1011.00', '2.a.1.b', 'active listening', '...', 'a636cb69257dcec699bce4f023a05126', 'active listening']
    ]
    with utils.makeNamedTemporaryCSV(content, '\t') as skills_filename:
        tagger = SimpleSkillTagger(
            skills_filename=skills_filename,
            hash_function=md5
        )
        result = [doc for doc in tagger.tagged_documents([
            'this is a job that needs active listening',
            'this is a reading comprehension job',
            'this is an active and reading listening job',
            'this is a reading comprehension and active listening job',
        ])]

        assert result == [
            'this is a job that needs a636cb69257dcec699bce4f023a05126',
            'this is a 2c77c703bd66e104c78b1392c3203362 job',
            'this is an active and reading listening job',
            'this is a 2c77c703bd66e104c78b1392c3203362 and a636cb69257dcec699bce4f023a05126 job',
        ]
