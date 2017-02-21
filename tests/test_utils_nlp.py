from utils import nlp

transforms = nlp.NLPTransforms()


def test_title_phase_one():
    assert transforms.title_phase_one('engineer') == 'engineer'
    assert transforms.title_phase_one('engineer/apply now') == 'engineer apply now'
    assert transforms.title_phase_one('engineer / apply now') == 'engineer apply now'
    assert transforms.title_phase_one("macy's engineer / apply now") == 'macys engineer apply now'
