"""Code for managing docstring parsing and conversion."""

from sphinx.ext.napoleon import Config

# Modify `sphinxcontrib.napoleon` to include `exputils` mod to Numpy style
# Use respective style: `sphinxcontrib.napoleon.GoogleDocstring(docstring,
# config)` to convert the docstring into ReStructuredText.
# Parse that restructured text to obtain the args, arg types, decsriptions,
# etc.

# TODO Docstring object that contains the parts of the docstring post parsing
class Docstring(object):
    """Docstring components.

    Attributes
    ----------
    style
    """
    def __init__(self)
        raise NotImplementedError()

class DocstringParser(object):
    """Docstring parsing.

    Attributes
    ----------
    """
    def __init__(self, style, doc_linking=False):
        raise NotImplementedError()

    def parse(docstring, style, doc_linking=False):
        """
        Args
        ----
        docstring : str
            docstring to be parsed
        doc_linking : str
            Linked docstring whose content applies to this docstring and will
            be parsed recursively.

        Returns
        -------
        Docstring
            A Docstring object containing the essential parts of the Docstring.
        """
        raise NotImplementedError()
        return Docstring()
