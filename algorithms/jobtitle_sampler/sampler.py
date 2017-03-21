import random

def get_num_lines():
    with open(filename) as f:
        num_lines = sum(1 for line in f)
    return num_lines

def reservoir_sample(sample_count, filename):
    random.seed(12345)
    sample_titles = []
    for index, line in enumerate(open(filename)):
        if index < sample_count:
            sample_titles.append(line)
        else:
            r = random.randint(0, index)
            if r < sample_count:
                sample_titles[r] = line
    return sample_titles



