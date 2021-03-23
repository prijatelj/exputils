"""Code for managing docstring parsing and conversion."""
from dataclasses import dataclass

from sphinx.ext.autodoc import getdoc, prepare_docstring
from sphinx.ext.napoleon import Config, GoogleDocstring, NumpyDocstring

# Modify `sphinxcontrib.napoleon` to include `exputils` mod to Numpy style
# Use respective style: `sphinxcontrib.napoleon.GoogleDocstring(docstring,
# config)` to convert the docstring into ReStructuredText.
# Parse that restructured text to obtain the args, arg types, decsriptions,
# etc.

# TODO Docstring object that contains the parts of the docstring post parsing

@dataclass
class Parameter(object):
    """Dataclass for parameters in docstrings."""

@dataclass
class Docstring(object):
    """Docstring components.

    Attributes
    ----------
    style
    """
    def __init__(self)
        raise NotImplementedError()

    def get_str(self, style):
        """Returns the docstring as a string in the given style.
        Args
        ----
        style : {'rst', 'google', 'numpy'}
            The style to output the Docstring's contents. `rst` is shorthand
            for reStructuredText,

        Returns
        -------
        str
            The contents of the Docstring instance as a string in the style.
        """
        raise NotImplementedError()


class DocstringParser(object):
    """Docstring parsing.

    Attributes
    ----------
    """
    def __init__(self, style, doc_linking=False):
        if (style := style.lower()) not in {'rst', 'numpy', 'google'}:
            raise ValueError(
                "Expected `style` = 'rst', 'numpy', or 'google', not `{style}`"
            )
        self.style = style
        self.doc_linking = doc_linking

        self.config = Config(napoleon_use_param=True, napoleon_use_rtype=True)
        raise NotImplementedError()

    def parse(docstring, style=None, doc_linking=False):
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
        docstring = prepare_docstring(docstring)

        style = self.style if style is None else style
        doc_linking = self.doc_linking if doc_linking is None else doc_linking

        # Ensure the docstring is in reStructuredText styling
        if self.style == 'google':
            docstring = GoogleDocstring(docstring, self.config)
        elif self.style == 'numpy':
            docstring = NumpyDocstring(docstring, self.config)

        # TODO parse the RST docstring

        # Find the Parameters/Arguments section(s)

        return Docstring()
