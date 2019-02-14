from nltk.tokenize import WordPunctTokenizer


def tf_sequence_from_candidate_skills(candidate_skills):
    posting_texts = {}
    cs_lookup = {}
    tokenizer = WordPunctTokenizer()
    for candidate_skill in candidate_skills:
        text_key = (
            candidate_skill.document_id,
            candidate_skill.skill_extractor_name
        )
        if text_key not in posting_texts:
            posting_texts[text_key] = candidate_skill.source_object['description']

        lookup_key = (
            text_key,
            candidate_skill.start_index
        )
        cs_lookup[lookup_key] = candidate_skill

    biglist = {
        'words': [],
        'tags': []
    }
    for text_key, text in posting_texts.items():
        words = []
        tags = []
        prior_cs = None
        for start_index, end_index in tokenizer.span_tokenize(text):
            word = text[start_index:end_index]
            print(word, start_index, end_index)
            words.append(word)
            if prior_cs:
                if end_index <= end_of_cs(prior_cs):
                    tags.append("I-SKILL")
                    if end_index == end_of_cs(prior_cs):
                        prior_cs = None
                else:
                    import ipdb
                    ipdb.set_trace()
                    raise ValueError(
                        f"overlap error, end index {end_of_cs(prior_cs)} of prior candidate skill '{prior_cs.skill_name}'" +
                        f" does not match up with token '{text[start_index:end_index]}'" +
                        f" that has range: {start_index} - {end_index}"
                    )
            elif (text_key, start_index) in cs_lookup:
                candidate_skill = cs_lookup[(text_key, start_index)]
                if end_index == end_of_cs(candidate_skill):
                    tags.append("S-SKILL")
                else:
                    tags.append("B-SKILL")
                    prior_cs = candidate_skill
            else:
                tags.append("O")

        biglist['words'].append(words)
        biglist['tags'].append(tags)
    return biglist


def end_of_cs(cs):
    return cs.start_index + len(cs.skill_name)
