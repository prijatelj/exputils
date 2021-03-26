"""Code for managing docstring parsing and conversion."""
from enum import Flag, unique
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from keyword import iskeyword
import re
from typing import NamedTuple

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


# Docstring object that contains the parts of the docstring post parsing
# TODO For each of these dataclasses, make them tokenizers for their respective
# parts of the docstring and string call them during parsing
@dataclass
class VariableDoc:
    """Dataclass for return objects in docstrings."""
    name : InitVar[str]
    description : str
    type : type = ValueExists.false

    def __post_init__(self, name):
        if name.isidentifier() and not iskeyword(name):
            self.name = name
        else:
            raise ValueError(f'`name` is an invalid variable name: `{name}`')


@dataclass
class ParameterDoc(VariableDoc):
    """Dataclass for parameters in docstrings."""
    default : InitVar[object] = ValueExists.false

    def __post_init__(self, default):
        # TODO generalize this by extending InitVar to perform this in init.
        # Default the "empty" value to None, possibly removing need to specify
        # `= None` in the dataclass part. Just to expedite this cuz it is now
        # repeated in this code, which itself is meant to expedite through
        # write once... Probably would not inherity from InitVar, but do
        # something similar-ish. Moreso generate the code snippet in the
        # generated __init__(). could possibly even reorder positional args
        # based on the order of inheritance.
        if default is None:
            raise TypeError(' '.join([
                'ParameterDoc() missing 1 required positional argument:',
                '`default` Provide via keyword, if able to through position.',
            ]))
        self.default = default


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
    short_description : InitVar[str] = None
    other_sections: OrderedDict({str : str}) = None

    def __post_init__(self, short_description):
        if short_description is None:
            raise TypeError(' '.join([
                'Docstring() missing 1 required positional argument:',
                '`short_description` Provide via keyword, if able to through',
                'position.',
            ]))
        self.short_description  = short_description


@dataclass
class FuncDocstring(Docstring):
    """Function docstring components.

    Attributes
    ----------
    short_description : str
        The short description of the docstring.
    args : str
        The function's Arguments/Parameters or the class' Attributes.
    """
    args : InitVar[OrderedDict({str : ParameterDoc})] = None
    other_args : OrderedDict({str : ParameterDoc}) = None # TODO implement parsing
    return_doc : VariableDoc = ValueExists.false

    def __post_init__(self, args):
        if args is None:
            raise TypeError(' '.join([
                'FuncDocstring() missing 1 required positional argument:',
                '`args` Provide via keyword, if able to through position.',
            ]))
        self.args  = args

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


@dataclass
class ClassDocstring(Docstring):
    """The multiple Docstrings the make up a class, including at least the
    class' docstring and the __init__ method's docstring.
    """
    attributes : InitVar[OrderedDict({str : ParameterDoc})] = None
    init_docstring : InitVar[FuncDocstring] = None
    methods : {str: FuncDocstring} = None

    def __post_init__(self, attributes, init_docstring):
        if attributes is None:
            raise TypeError(' '.join([
                'ClassDocstring() missing 1 required positional argument:',
                '`attributes` Provide via keyword, if able to through',
                'position.',
            ]))
        self.attributes  = attributes

        if init_docstring is None:
            raise TypeError(' '.join([
                'ClassDocstring() missing 1 required positional argument:',
                '`init_docstring` Provide via keyword, if able to through',
                'position.',
            ]))
        self.init_docstring  = init_docstring


class DocstringParser(object):
    """Docstring parsing.

    Attributes
    ----------
    style : {'rst', 'numpy', 'google'}
        The style expected to parse.
    doc_linking : bool = False
    config : sphinx.ext.napoleon.Config = None
    """
    def __init__(self, style, doc_linking=False, config=None):
        if (style := style.lower()) not in {'rst', 'numpy', 'google'}:
            raise ValueError(
                "Expected `style` = 'rst', 'numpy', or 'google', not `{style}`"
            )
        # TODO allow the passing of a func/callable to transform custom doc
        # styles, thus allowing users to easily adapt to their doc style
        self.style = style
        self.doc_linking = doc_linking

        if config is None:
            self.config = Config(
                napoleon_use_param=True,
                napoleon_use_rtype=True,
            )
        else:
            self.config = config

        # TODO section tokenizer (then be able to tell if the section is one of
        #   param/arg or type and to pair those together.
        #   attribute sections `.. attribute:: attribute_name`

        # Regexes for RST field and directive, and together for sections
        self.re_field = re.compile(':[ \t]*(\w[ \t]*)+:')
        self.re_directive = re.compile('\.\.[ \t]*(\w[ \t]*)+:')
        self.re_section = re.compile(
            '|'.join([self.re_field.pattern, self.re_directive])
        )

        # Temporary useful patterns
        section_end_pattern = f'({self.re_section.pattern}|\Z)'
        name_pattern = '[ \t]+(?P<name>[\*\w]+)'
        doc_pattern = f'[ \t]*(?P<doc>.*?){section_end_pattern}'

        # Regexes for checking and parsing the types of sections
        self.re_param = re.compile(f':param{name_pattern}:{doc_pattern}', re.S)
        self.re_type = re.compile(f':type{name_pattern}:{doc_pattern}', re.S)

        self.re_returns = re.compile(':returns:{doc_pattern}', re.S)
        self.re_returns = re.compile(':rtype:{doc_pattern}', re.S)

        self.re_attribute = re.compile(
            f'\.\.[ \t]*attribute[ \t]*::{name_pattern}[ \t]*\n{doc_pattern}',
            re.S,
        )
        # TODO extract attribute type ':type: <value>\n'

        #self.re_other_params = re.compile()

        self.re_rubric = re.compile(
            f'\.\.[ \t]*attribute[ \t]*::{name_pattern}[ \t]*\n{doc_pattern}',
            re.S,
        )

    def parse_func(self, docstring, name, obj_type):
        """Parse the docstring of a function."""
        docstring = prepare_docstring(docstring)

        # NOTE for now, use what is specified at initialization
        #style = self.style if style is None else style.lower()
        #doc_linking = self.doc_linking if doc_linking is None else doc_linking

        # Convert the docstring is in reStructuredText styling
        if self.style == 'google':
            docstring = GoogleDocstring(docstring, self.config)
        elif self.style == 'numpy':
            docstring = NumpyDocstring(docstring, self.config)
        # TODO allow the passing of a func/callable to transform custom doc
        # styles

        # TODO parse the RST docstring
        #   TODO find Parameters/Args and other Param like sections.
        #   TODO Get all param names, their types (default to object w/ logging
        #   of no set type), and their descriptions

        # Prepare the paired sets for parameters to catch dups, & missing pairs
        param_set = set()
        type_set = set()

        if len(docstring) < 1:
            raise ValueError('The docstring is only a short description!')

        # Short description is always the first line only
        short_description = docstring[0]

        # TODO Get the start indices of any & all sections (Could parallelize)
        section_start_indices = []
        # TODO this would be way nicer with a kind of top down tokenization
        # where each token then can be further broken down, so Section, then
        # the section type, then its parts. This is where the above dataclasses
        # having their own parser functions for RST would be good. This would
        # probably involve back tracking tho and multiple passes.
        for i, line in enumerate(docstring[1:], start=1):
            if self.re_section.match(line):
                section_start_indices.append(i)

        # Long description initial text till next section or end of str.
        # TODO consider smarter combine to avoid forced column lengths in desc
        long_description = '\n'.join(docstring[:section_start_indices[0]])

        num_sections = len(section_start_indices)
        if num_sections < 1:
            raise ValueError('The given docstring includes no sections.')
        elif num_sections == 1:
            # TODO if 1 section, then check if params, ow. error
            if not (param_parsed := self.re_param.match()):
                raise ValueError(
                    'The docstring does not include a parameters section!'
                )
            # TODO do something with param_parsed, and make param regex extract
            # the name and doc.
        else:
            # TODO check first section and for rest loop thru the sections by
            # [start_idx:end] where end is the current itr and prior updates to
            # this sections beginning.

            # Add the end value so a check is unnecessary repetitively
            section_start_indices.append(len(docstring))

            param = []
            types = []
            returns = None
            rtype = None
            for i, line in enumerate(section_start_indices[1:], start=1):
                # TODO parse the section between docstring[i - 1:i]
                if section_parsed := self.re_param.match(line):
                    param_indices.append(i)
                elif section_parsed := self.re_type.match(line):
                    param_indices.append(i)
                elif section_parsed := self.attribute.match(line):
                    attr_indices.append(i)
                elif section_parsed := self.re_returns.match(line):
                    if returns is None:
                        returns = PARSE_RETURNS_SECTION
                    else:
                        raise SyntaxError(' '.join([
                            'The docstring cannot have more than one',
                            '`returns` section.',
                        ]))
                elif section_parsed := self.re_rtype.match(line):
                    if rtype is None:
                        rtype = PARSE_RTYPE_SECTION
                    else:
                        raise SyntaxError(' '.join([
                            'The docstring cannot have more than one `rtype`',
                            'section.',
                        ]))
                # else: Some other section, save as {str : str} in other_sectios

        # Rejoin lines into one str for regex. This strikes me as inefficient
        #docstring = '\n'.join(docstring)

        # Return the Docstring Object that stores that docsting info
        raise NotImplemented()

    def parse(
        obj,
        style=None,
        doc_linking=False,
        name=None,
        obj_type=None,
        func_list=None
    ):
        """
        Args
        ----
        obj : object | str
            The object whose __doc__ is to be parsed. If a str object of the
            docstring, then the `name` and `obj_type` parameters must be
            provided.
        style : str, default None
            Expected docstring style determining how to parse the docstring.
        doc_linking : bool, default False
            Linked docstring whose content applies to this docstring and will
            be parsed recursively.

            TODO Add whitelisting of which packages to allow recursive doc
            linking
        name : str, optional
            The name of the object whose docstring is being parsed. Only needs
            to be supplied when the `obj` is a `str` of the docstring to be
            parsed, otherwies not used.
        obj_type : type, optional
            The type of the object whose docstring is being parsed. Only needs
            to be supplied when the `obj` is a `str` of the docstring to be
            parsed, otherwies not used.
        func_list : [str], optional
            A list of method names whose docstrings are all to be parsed and
            returned in a hierarchical manner encapsulating the hierarchical
            docstring of a class.  Only used when `obj` is a class object.

        Returns
        -------
        FuncDocstring | ClassDocstring
            A Docstring object containing the essential parts of the Docstring,
            or a ClassDocstring which contains the Docstring objects of the
            methods of the class along with the class' parsed Docstring object.
        """
        # TODO optionally parallelize the parsing, perhaps with Ray?
        if isinstance(obj, str):
            if name is None or obj_type is None:
                raise ValueError(' '.join([
                    'If `obj` is a `str` then must also give `name` and',
                    '`obj_type`',
                ]))
            return self.parse_func(obj, name, obj_type)
        elif hasattr(obj, '__doc__') and hasattr(obj, '__name__'):
            # Obtain info from the object itself
            name = obj.__name__
            obj_type = type(obj)

            # TODO if a class, then parse the __init__ too, but as the main
            # docstring of interest. As it defines the params to give, and thus
            # the args we care about.
            if isinstance(obj_type, type):
                class_docstring = obj.__doc__
                init_docstring = self.parse_func(obj.__init__)

                raise NotImplementedError('Need to add parsing of a class')
                # TODO add parsing of a class' methods in given list.
                return ClassDocstring(name, obj_type, description, )

            return self.parse_func(obj.__doc__, name, obj_type)
        else:
            raise TypeError(' '.join([
                'Expected `obj` to be an object with `__doc__` and `__name__`',
                'attributes, or a `str` with the `name` and `obj_type`',
                'parameters given.',
            ]))
