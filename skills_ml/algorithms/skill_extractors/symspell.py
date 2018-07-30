# SymSpell algorithm
# Taken from https://github.com/ne3x7/pysymspell/ as of 2018-07-24
# and modified to make the input format match skills-ml's expectations more closely
# SymSpell original copyright:
#
# MIT License
#
# Copyright (c) 2018 Nikolay Stulov (Python Port https://github.com/ne3x7/pysymspell)
# Copyright (c) 2018 Hampus Londögård (Java Port https://github.com/Lundez/JavaSymSpell)
# Copyright (c) 2018 Wolf Garbe (Original C# implementation https://github.com/wolfgarbe/SymSpell)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# https://opensource.org/licenses/MIT



import numpy as np
import os
import sys

class SymSpell():
    """SymSpell: 1 million times faster through Symmetric Delete spelling correction algorithm.
    The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and dictionary lookup
    for a given Damerau-Levenshtein distance. It is six orders of magnitude faster and language independent.
    Opposite to other algorithms only deletes are required, no transposes + replaces + inserts.
    Transposes + replaces + inserts of the input term are transformed into deletes of the dictionary term.
    Replaces and inserts are expensive and language dependent: e.g. Chinese has 70,000 Unicode Han characters!
    SymSpell supports compound splitting / decompounding of multi-word input strings with three cases:
    1. mistakenly inserted space into a correct word led to two incorrect terms 
    2. mistakenly omitted space between two correct words led to one incorrect combined term
    3. multiple independent input terms with/without spelling errors
    See https://github.com/wolfgarbe/SymSpell for details.
    Args:
        max_dictionary_edit_distance (int, optional): Maximum distance used to generate index. Also acts 
            as an upper bound for `max_edit_distance` parameter in `lookup()` method. Defaults to 2.
        prefix_length (int, optional): Prefix length. Should not be changed normally. Defaults to 7.
        count_threshold (int, optional): Threshold corpus-count value for words to be considered correct.
            Defaults to 1, values below zero are also mapped to 1. Consider setting a higher value if your
            corpus contains mistakes.
    """
    def __init__(self, max_dictionary_edit_distance=2, prefix_length=7, count_threshold=1):
        self._distance_algorithm = 'damerau'
        self._max_length = 0
        self._deletes = None
        self._words = None
        self._below_threshold_words = dict()

        if max_dictionary_edit_distance < 0:
            max_dictionary_edit_distance = 2
        if prefix_length < 1 or prefix_length <= max_dictionary_edit_distance:
            prefix_length = 7
        if count_threshold < 0:
            count_threshold = 1

        self._words = dict()
        self._max_dictionary_edit_distance = max_dictionary_edit_distance
        self._prefix_length = prefix_length
        self._count_threshold = count_threshold
        self._compact_mask = (0xffffffff >> 8) << 2

    def _create_dictionary_entry(self, key, count):
        """Creates or updates a dictionary entry.
        Args:
            key (str): Word to insert or update.
            count (int): Count to save or add to existing.
        Returns:
            bool: True if word was added to the dictionary, False if word was updated or ignored.
        """
        if count <= 0:
            if self._count_threshold > 0:
                return False
            count = 0

        if self._count_threshold > 1 and key in self._below_threshold_words.keys():
            count_previous = self._below_threshold_words[key]
            count = count_previous + count
            if count >= self._count_threshold:
                self._below_threshold_words.pop(key)
            else:
                self._below_threshold_words[key] = count
                return False
        elif key in self._words.keys():
            count_previous = self._words[key]
            count = count_previous + count
            self._words[key] = count
            return False
        elif count < self._count_threshold:
            self._below_threshold_words[key] = count
            return False

        self._words[key] = count
        if len(key) > self._max_length:
            self._max_length = len(key)

        edits = self._edits_prefix(key)

        if self._deletes is None:
            self._deletes = dict()

        for edit in edits:
            hs = self._hash(edit)
            suggestions = list()
            if hs in self._deletes.keys():
                suggestions = self._deletes.get(hs)
                suggestions.append(key)
                self._deletes[hs] = suggestions
            else:
                suggestions = [key]
                self._deletes[hs] = suggestions
        return True

    def load_dictionary(self, corpus):
        """Loads dictionary from :param:`corpus` file.
        File should contain space-separated word-count pairs one at a line.
        Args:
            corpus (str): Path to corpus file.
        """
        if os.path.exists(corpus):
            with open(corpus, 'r') as f:
                for line in f:
                    key, count = line.split()
                    count = int(count)
                    self._create_dictionary_entry(key, count)

        if self._deletes is None:
            self._deletes = dict()

    def create_dictionary(self, corpus):
        """Creates dictionary from :param:`corpus` file.
        Note:
            Words are not preprocessed in any way. It is your duty to provide appropriate corpus. Also
                keep in mind that the distance used to generate index is specified at initialization. Consider
                doing a purge of below threshold words afterwards.
        Args:
            corpus (str): Path to corpus file.
        """
        for term in corpus:
            self._create_dictionary_entry(term, 1)

        if self._deletes is None:
            self._deletes = dict()

    def purge_below_threshold_words(self):
        """Purges words below threshold.
        Consider using this method after creating a dictionary to reduce memory usage. These words are not
            used in any way during lookup.
        """
        self._below_threshold_words = dict()

    def lookup(self, phrase, verbosity, max_edit_distance):
        """Attempts to correct the spelling of :param:`phrase`.
        Note:
            Phrase is not preprocessed in any way.
        Args:
            phrase (str): Word to correct. Should be a valid word.
            verbosity (int, 0, 1 or 2): Output toggle. Set to 0 to output closest most common correction,
                set to 1 to output closest suggestion, set to 2 to output all suggestions.
            max_edit_distance (int): Maximum edit distance to consider.
        Returns:
            list of :obj:`SuggestionItem`: Suggested corrections.
        Raises:
            AssertionError: If :param:`max_edit_distance` is larger than maximum edit distance specified 
                at initialization.
        """
        assert max_edit_distance <= self._max_dictionary_edit_distance, 'Distance too big'
        suggestions = list()
        phrase_len = len(phrase)

        if phrase_len - max_edit_distance > self._max_length:
            return suggestions

        considered_deletes = set()
        considered_suggestions = set()
        suggestions_count = 0

        if phrase in self._words.keys():
            suggestions_count = self._words[phrase]
            suggestions.append(SuggestionItem(phrase, 0, suggestions_count))
            if verbosity < 2:
                return suggestions
        considered_suggestions.add(phrase)

        max_edit_distance_2 = max_edit_distance
        candidate_ptr = 0
        candidates = list()

        phrase_prefix_len = phrase_len
        if phrase_prefix_len > self._prefix_length:
            phrase_prefix_len = self._prefix_length
            candidates.append(phrase[:phrase_prefix_len])
        else:
            candidates.append(phrase)

        comp = EditDistance(phrase, self._distance_algorithm)

        while candidate_ptr < len(candidates):
            candidate = candidates[candidate_ptr]
            candidate_ptr += 1
            candidate_len = len(candidate)
            len_diff = phrase_prefix_len - candidate_len

            if len_diff > max_edit_distance_2:
                if verbosity == 2:
                    continue
                else:
                    break

            if self._hash(candidate) in self._deletes.keys():
                dictionary_suggestions = self._deletes[self._hash(candidate)]
                for suggestion in dictionary_suggestions:
                    if suggestion == phrase:
                        continue
                    suggestion_len = len(suggestion)

                    if abs(suggestion_len - phrase_len) > max_edit_distance_2 \
                            or suggestion_len < candidate_len \
                            or (suggestion_len == candidate_len and suggestion != candidate):
                        continue

                    suggestion_prefix_len = min(suggestion_len, self._prefix_length)
                    if suggestion_prefix_len > phrase_prefix_len and suggestion_prefix_len - candidate_len > max_edit_distance_2:
                        continue

                    distance = np.inf
                    min_distance = 0

                    if candidate_len == 0:
                        distance = max(phrase_len, suggestion_len)
                        if distance > max_edit_distance_2 or suggestion in considered_suggestions:
                            considered_suggestions.add(suggestion)
                            continue
                        considered_suggestions.add(suggestion)
                    elif suggestion_len == 1:
                        if phrase.index(suggestion[0]) < 0:
                            distance = phrase_len
                        else:
                            distance = phrase_len - 1
                        if distance > max_edit_distance_2 or suggestion in considered_suggestions:
                            considered_suggestions.add(suggestion)
                            continue
                        considered_suggestions.add(suggestion)
                    else:
                        min_distance = min(phrase_len, suggestion_len) - self._prefix_length
                        if self._prefix_length - max_edit_distance == candidate_len \
                            and (min_distance > 1 and phrase[:phrase_len + 1 - min_distance] != suggestion[:phrase_len + 1 - min_distance]) \
                            or (min_distance > 0 and phrase[phrase_len - min_distance] != suggestion[suggestion_len - min_distance] \
                                and (phrase[phrase_len - min_distance - 1] != suggestion[suggestion_len - min_distance] \
                                or phrase[phrase_len - min_distance] != suggestion[suggestion_len - min_distance - 1])):
                            continue
                        else:
                            if (verbosity < 2 and not self._delete_in_suggestion_prefix(candidate, candidate_len, suggestion, suggestion_len)) \
                                or suggestion in considered_suggestions:
                                considered_suggestions.add(suggestion)
                                continue
                            considered_suggestions.add(suggestion)
                            distance = comp.compare(suggestion, max_edit_distance_2)
                            if distance < 0:
                                continue

                    if distance <= max_edit_distance_2:
                        suggestion_count = self._words[suggestion]
                        si = SuggestionItem(suggestion, distance, suggestion_count)
                        if len(suggestions) > 0:
                            if verbosity == 0: # Top
                                if distance < max_edit_distance_2 or suggestion_count > suggestions[0].count:
                                    max_edit_distance_2 = distance
                                    suggestions[0] = si
                                continue
                            elif verbosity == 1: # Closest
                                if distance < max_edit_distance_2:
                                    suggestions = list()
                                break
                        if verbosity < 2:
                            max_edit_distance_2 = distance
                        suggestions.append(si)

            if len_diff < max_edit_distance and candidate_len <= self._prefix_length:
                if verbosity < 2 and len_diff >= max_edit_distance_2:
                    continue

                for i in range(candidate_len):
                    d = candidate[:i] + candidate[i+1:]

                    if d not in considered_deletes:
                        considered_deletes.add(d)
                        candidates.append(d)

        if len(suggestions) > 1:
            suggestions.sort()
        return suggestions

    def lookup_compound(self, phrase, max_edit_distance):
        """Attempts to correct the spelling of :param:`phrase`.
        Note:
            Phrase is not preprocessed in any way.
        Args:
            phrase (str): Sentence to correct.
            max_edit_distance (int): Maximum edit distance to consider for each word.
        Returns:
            list of :obj:`SuggestionItem`: Length-one list with suggested correction.
        Raises:
            AssertionError: If :param:`max_edit_distance` is larger than maximum edit distance specified 
                at initialization.
        """
        assert max_edit_distance <= self._max_dictionary_edit_distance, 'Distance too big'

        terms_list_1 = self._parse_words(phrase)
        suggestions = list()
        suggestions_parts = list()
        suggestions_combi = list()
        edit_distance = np.inf

        last_combi = False
        for i in range(len(terms_list_1)):
            suggestions = self.lookup(terms_list_1[i], 0, max_edit_distance)
            if i > 0 and not last_combi:
                suggestions_combi = self.lookup(terms_list_1[i-1] + terms_list_1[i], 0, max_edit_distance)
                if len(suggestions_combi) > 0:
                    best_1 = suggestions_parts[-1]
                    if len(suggestions) > 0:
                        best_2 = suggestions[0]
                    else:
                        best_2 = SuggestionItem(terms_list_1[i], max_edit_distance+1, 0)

                    distance = EditDistance(terms_list_1[i-1] + ' ' + terms_list_1[i], 'damerau')
                    if suggestions_combi[0].distance + 1 < distance.damerau_levenshtein_distance(best_1.term + ' ' + best_2.term, max_edit_distance):
                        suggestions_combi[0].distance += 1
                        suggestions_parts[-1] = suggestions_combi[0]
                        last_combi = True
                        continue

            last_combi = False

            if len(suggestions) > 0 and (suggestions[0].distance == 0 or len(terms_list_1[i]) == 1):
                suggestions_parts.append(suggestions[0])
            else:
                suggestions_split = list()

                if len(suggestions) > 0:
                    suggestions_split.append(suggestions[0])

                if len(terms_list_1[i]) > 1:
                    for j in range(1, len(terms_list_1[i])):
                        part_1 = terms_list_1[i][:j]
                        part_2 = terms_list_1[i][j:]
                        suggestions_1 = self.lookup(part_1, 1, max_edit_distance)

                        if len(suggestions_1) > 0:
                            if len(suggestions) > 0 and suggestions[0] == suggestions_1[0]:
                                continue
                            suggestions_2 = self.lookup(part_2, 1, max_edit_distance)

                            if len(suggestions_2) > 0:
                                if len(suggestions) > 0 and suggestions[0] == suggestions_2[0]:
                                    continue

                                split = suggestions_1[0].term + ' ' + suggestions_2[0].term
                                edit_distance = EditDistance(terms_list_1[i], 'damerau')
                                suggestion_split = SuggestionItem(split, 
                                    edit_distance.damerau_levenshtein_distance(split, max_edit_distance), 
                                    min(len(suggestions_1), len(suggestions_2)))
                                if suggestion_split.distance >= 0:
                                    suggestions_split.append(suggestion_split)
                                if suggestion_split.distance == 1:
                                    break

                    if len(suggestions_split) > 0:
                        suggestions_split.sort()
                        suggestions_parts.append(suggestions_split[0])
                    else:
                        si = SuggestionItem(terms_list_1[i], 0, max_edit_distance + 1)
                        suggestions_parts.append(si)

                else:
                    si = SuggestionItem(terms_list_1[i], 0, max_edit_distance + 1)
                    suggestions_parts.append(si)

        suggestion = SuggestionItem('', sys.maxsize, sys.maxsize)
        s = ' '.join([x.term for x in suggestions_parts])
        suggestion.count = min([x.count for x in suggestions_parts])
        suggestion.term = s
        edit_distance = EditDistance(suggestion.term, 'damerau')
        suggestion.distance = edit_distance.damerau_levenshtein_distance(phrase, self._max_dictionary_edit_distance)
        return [suggestion]

    def _delete_in_suggestion_prefix(self, delete, delete_len, suggestion, suggestion_len):
        """Helper method to check if :param:`delete` is prefix of :param:`suggestion`.
        Args:
            delete (str): String to look for in prefix.
            delete_len (int): Length of :param:`delete`.
            suggestion (str): String to take prefix from.
            suggestion_len (int): Length of :param:`suggestion`.
        Returns:
            bool: True if :param:`delete` is prefix of :param:`suggestion`, False otherwise.
        """
        if delete_len == 0:
            return True
        if self._prefix_length < suggestion_len:
            suggestion_len = self._prefix_length

        j = 0
        for i in range(delete_len):
            ch = delete[i]

            while j < suggestion_len and ch != suggestion[j]:
                j += 1

            if j == suggestion_len:
                return False
        return True

    def _edits(self, word, edit_distance, delete_words):
        """helper recursive method to generate deletes.
        Refer to article for details.
        Args:
            word (str): Word to generate deletes from.
            edit_distance (int): Maximum edit distance to consider, recursion depth.
            delete_words (set): Generated deletes, pass empty set first time.
        Returns:
            delete_words (set): Generated deletes.
        """
        edit_distance += 1
        for i in range(len(word)):
            delete = word[:i] + word[i+1:]
            if delete not in delete_words:
                delete_words.add(delete)
                if edit_distance < self._max_dictionary_edit_distance:
                    self._edits(delete, edit_distance, delete_words)
        return delete_words

    def _edits_prefix(self, key):
        s = set()
        if len(key) <= self._max_dictionary_edit_distance:
            s.add('')
        if len(key) > self._prefix_length:
            key = key[:self._prefix_length]
        s.add(key)
        return self._edits(key, 0, s)

    def _hash(self, s):
        l = len(s)
        l_mask = l
        if l_mask > 3:
            l_mask = 3

        hs = 2166136261
        for i in range(l):
            hs ^= ord(s[i])
            hs *= 16777619
        hs &= self._compact_mask
        hs |= l_mask
        return hs

    def _parse_words(self, text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
        if lower:
            text = text.lower()

        translate_map = str.maketrans(filters, split * len(filters))

        text = text.translate(translate_map)
        seq = text.split(split)
        return [i for i in seq if i]

class EditDistance():
    def __init__(self, base_string, distance_algorithm):
        self._base_string = base_string
        self._distance_algorithm = distance_algorithm
        if self._base_string == '':
            self._base_string = None
            return
        if self._distance_algorithm == 'damerau':
            self._v0 = np.zeros(len(self._base_string), dtype=np.int32)
            self._v2 = np.zeros(len(self._base_string), dtype=np.int32)

    def compare(self, string_2, max_distance):
        if self._distance_algorithm == 'damerau':
            return self.damerau_levenshtein_distance(string_2, max_distance)

    def damerau_levenshtein_distance(self, string_2, max_distance):
        if self._base_string is None:
            if string_2 is None:
                return 0
            else:
                return len(string_2)

        if string_2 is None or string_2 == '':
            return len(self._base_string)

        if len(self._base_string) > len(string_2):
            string_1 = string_2
            string_2 = self._base_string
        else:
            string_1 = self._base_string

        slen = len(string_1)
        tlen = len(string_2)

        while slen > 0 and string_1[slen-1] == string_2[tlen-1]:
            slen -= 1
            tlen -= 1

        st = 0
        if string_1[0] == string_2[0] or slen == 0:
            while st < slen and string_1[st] == string_2[st]:
                st += 1

            slen -= st
            tlen -= st

            if slen == 0:
                return tlen

            string_2 = string_2[st:st+tlen]

        len_diff = tlen - slen
        if max_distance < 0 or max_distance > tlen:
            max_distance = tlen
        elif len_diff > max_distance:
            return -1

        if tlen > self._v0.shape[0]:
            self._v0 = np.zeros(tlen, dtype=np.int32)
            self._v2 = np.zeros(tlen, dtype=np.int32)
        else:
            for i in range(tlen):
                self._v2[i] = 0

        for j in range(max_distance):
            self._v0[j] = j + 1

        for j in range(max_distance, tlen):
            self._v0[j] = max_distance + 1

        j_st_offset = max_distance - (tlen - slen)
        have_max = max_distance < tlen
        j_st = 0
        j_fn = max_distance
        s_ch = string_1[0]
        cur = 0
        for i in range(slen):
            prev_s_ch = s_ch
            s_ch = string_1[st + i]
            t_ch = string_2[0]
            left = i
            cur = left + 1
            next_trans_cost = 0
            if i > j_st_offset:
                j_st += 1
            if j_fn < tlen:
                j_fn += 1
            for j in range(j_st, j_fn):
                above = cur
                this_trans_cost = next_trans_cost
                next_trans_cost = self._v2[j]
                cur = left
                self._v2[j] = cur
                left = self._v0[j]
                prev_t_ch = t_ch
                t_ch = string_2[j]
                if s_ch != t_ch:
                    if left < cur:
                        cur = left
                    if above < cur:
                        cur = above
                    cur += 1
                    if i != 0 and j != 0 and s_ch == prev_t_ch and prev_s_ch == t_ch:
                        this_trans_cost += 1
                        if this_trans_cost < cur:
                            cur = this_trans_cost
                self._v0[j] = cur
            if have_max and self._v0[i + len_diff] > max_distance:
                return -1
        if cur <= max_distance:
            return cur
        else:
            return -1

class SuggestionItem():
    def __init__(self, term, distance, count):
        self._term = term
        self._distance = distance
        self._count = count

    def __eq__(self, other):
        if self._distance == other._distance:
            return self._count == other._count
        else:
            return self._distance == other._distance

    def __lt__(self, other):
        if self._distance == other._distance:
            return self._count > other._count
        else:
            return self._distance < other._distance

    def __str__(self):
        return self._term + ':' + str(self._count) + ':' + str(self._distance)

    def __repr__(self):
        return self._term + ':' + str(self._count) + ':' + str(self._distance)

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, count):
        self._count = count

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        self._distance = distance

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, term):
        self._term = term
