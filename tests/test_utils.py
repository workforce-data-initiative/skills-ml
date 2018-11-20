from skills_ml.utils import itershuffle

def test_itershuffle():
    data = range(100)

    shuffled = list(itershuffle(data))

    assert len(data) == len(shuffled)
    assert set(data) == set(shuffled)
    assert data != list(shuffled)
