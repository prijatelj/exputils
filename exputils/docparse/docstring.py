"""Code for managing docstring parsing and conversion."""
from enum import Flag, unique
from dataclasses import dataclass, InitVar
from keyword import iskeyword

from sphinx.ext.autodoc import getdoc, prepare_docstring
from sphinx.ext.napoleon import Config, GoogleDocstring, NumpyDocstring

# Modify `sphinxcontrib.napoleon` to include `exputils` mod to Numpy style
# Use respective style: `sphinxcontrib.napoleon.GoogleDocstring(docstring,
# config)` to convert the docstring into ReStructuredText.
# Parse that restructured text to obtain the args, arg types, decsriptions,
# etc.

# TODO, I need the first working version of this done and while it seems like
# it would make the most sense to use and/or modify the sphinx DocTree of
# classes and functions to manage the OOP of docstrings, the sphinx docs and
# code upon my 3 attempts at reading and understanding did not yeild a
# sufficient enough understanding to be able to implement this probably more
# desirable approach. I think, if it is posisble to use the sphinx DocTree of
# classes and functions within the code itself, it would be best to rely on
# this pre-existing backbone and avoid redundancy.

# TODO, uncertain if this already exists, e.g. in Sphinx, but would be nice if
# we could go from the OOP representation of a docstring to the actual str
# easily, and swap styles. Probably similar to how Google and Numpy docs are
# parsed, where they are converted to reStructuredText and then parsed. So,
# output Docstring Object to str of RST and then optionally convert to Google
# or Numpy.

@unique
class ValueExists(Flag):
    """Enum for standing in for a non-existent default value."""
    true = True
    false = False


# TODO Docstring object that contains the parts of the docstring post parsing
@dataclass
class VariableDoc(object):
    """Dataclass for return objects in docstrings."""
    name : InitVar[str]
    type : type = ValueExists.false
    description : str

    def __post_init__(self, name):
        if name.isidentifier() and not iskeyword(name):
            self.name = name
        else:
            raise ValueError(f'`name` is an invalid variable name: `{name}`')

@dataclass
class Parameter(VariableDoc):
    """Dataclass for parameters in docstrings."""
    default : object = ValueExists.false

@dataclass
class Docstring(VariableDoc):
    """Docstring components.

    Attributes
    ----------
    short_description : str
        The short description of the docstring.
    args : str
        The function's Arguments/Parameters or the class' Attributes.
    """
    short_description : str
    args : {str : ParameterDoc}
    return_doc : InitVar[VariableDoc]
    # TODO general sections: {section_name : str}

    def __post_init__(self, args, return_doc):
        raise NotImplementedError()

    def get_str(self, style):
        """Returns the docstring as a string in the given style. Could simply
        be __str__() and always return RST, and make available RST to
        Google/Numpy functions/classes.

        Args
        ----
        style : {'rst', 'google', 'numpy'}
            The style to output the Docstring's contents. `rst` is shorthand
            for reStructuredText.

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
    style : {'rst', 'numpy', 'google'}
        The style expected to parse.
    doc_linking : bool, default False
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
        doc_linking : bool, default False
            Linked docstring whose content applies to this docstring and will
            be parsed recursively.

            Add whitelisting of which packages to allow recursive doc linking

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
        #   TODO get short description, long description.
        #   TODO find Parameters/Args and other Param like sections.
        #   TODO Get all param names, their types (default to object w/ logging
        #   of no set type), and their descriptions

        # Return the Docstring Object that stores that docsting ingo
        return Docstring()
