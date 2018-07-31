from skills_ml.algorithms.string_cleaners.nlp import NLPTransforms, deep

transforms = NLPTransforms()


def test_title_phase_one():
    assert transforms.title_phase_one('engineer') == 'engineer'
    assert transforms.title_phase_one('engineer/apply now') == 'engineer apply now'
    assert transforms.title_phase_one('engineer / apply now') == 'engineer apply now'
    assert transforms.title_phase_one("macy's engineer / apply now") == 'macys engineer apply now'
    assert transforms.clean_html("<h1>apply now <p>engineer</p></h1>") == 'apply now engineer'


def test_deep_wrapper():
    assert deep(transforms.clean_str)([["macy's engineer / apply now", "engineer/apply now"], ["engineer.", "python!"]]) == \
        [["macy s engineer apply now", "engineer apply now"], ["engineer ", "python "]]


