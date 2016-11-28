class SkillTaggerBase(object):
    """
    A base class for objects that tag skills in job listing data
    """
    def __init__(self, skills_filename):
        """
        Args:
            skills_filename: The name of a file that contains skills data
        """
        self.skills_filename = skills_filename

    def _label_titles(self, corpus_generator):
        """
        Args:
            corpus_generator: iterable that yields job listings
        Returns:
            iterable with tagged
        """
        return []

    def tagged_documents(self, documents):
        for document in self._label_titles(documents):
            yield document
