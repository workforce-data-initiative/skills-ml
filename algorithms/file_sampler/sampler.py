import random

def get_num_lines():
    with open(filename) as f:
        num_lines = sum(1 for line in f)
    return num_lines

def reservoir_sample(sample_count, filename, rand_seed):
    """
    Sample rows from a csv file
    Args:
        sample_count: the number of sampled data points we want
        filename: csv file directory
        rand_seed: random seed
    Returns:
        A list of sample row of the csv file
    """
    random.seed(rand_seed)
    sample_titles = []
    for index, line in enumerate(open(filename)):
        if index < sample_count:
            sample_titles.append(line)
        else:
            r = random.randint(0, index)
            if r < sample_count:
                sample_titles[r] = line
    return sample_titles



