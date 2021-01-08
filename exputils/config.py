"""Create argparse compatible configuration objects of objects in Python as
determined by their docstring.

Notes
-----
This probably is better served as its own package once more complete. It is
meta-programming to expedite the coding process of configuration files and
argparsers based on the docstring. It could also include decorators for
"auto-completion" or "meta-completion" of code by parsing the docstring and
setting the function's, class' or callable's positional args, and keyword args.
Due to the desire of wanting to expedite the coding process, it follows the
design principle of "write once", meaning to write something once and use many
times. If support of main docstring types are included by the parser, then it
may even be possible to obtain a config and arg parser for 3rd party code.

This relies on the docstring being written correctly within it's docstring
style, which then mean the use of this code, encourages properly written doc
strings, similar to how python inherently requires properly indented code. This
also being optional, is of course a plus where a lack of complete docstrings is
desired.

The meta-completion portion could possibly include a "compile" feature that
creates the resulting python code to avoid computation costs of using the
decorator to parse the function's docstring everytime. And type checking could
be optional, ofc, and possibly even include custom functions for type checking
certain types.

This would benefit from also including a docstring conversion and underlying
default state information for docstrings to ease translation between
docstrings. I suppose, this could be used in a way like yapf is to organize
docstrings within a desired stylistic parameters that do not affect
functionality of the parsing.

Parts coming into play: config file creation/parsing, argparser creation, type
checking, meta-completion, style check/correction.

Profiling from CLI: it would be beneficial if the CLI had the ability to turn
on profiling for the different configs for certain objects, this way those
objects in particular would be profiles when their individual parts ran. This
would require a decorator to be able to give the CLI control over the function,
or around whenevr the function or class is executed.
    Separate loggers possible to this way, if so desired.
"""
import argparse
import re

# TODO make regex for the different parts so the styles just supply their own
# regex, rather than needing code for them.

# TODO create the function when given any function or class, takes the
# docstring, optionally checks if docstring roughly matches class/args to avoid
# issues mismatching docstring to object, and optionally creates the argparse
# parser for that object using the information from the docstring.

class DocStringConfig(object):
    """Create argparse compatible configuration objects from objects in Python
    as determined by their docstring (`__doc__`).

    Attributes
    ----------
    style : {'numpy', 'google', 'exputils'}
        The docstring style that is expected to be parsed.
    desc : {'paragraph', 'short', 'long'}, optional
        The expected format for the description of argparser parsers'.
    desc_re : re.Pattern
        Regex pattern for finding the description within the docstring. The
        description is typically the short summary of the docstring plus any
        text that follows if there is no new line following the short summary,
        i.e. a longer paragraph is used in conjunction with the short summary,
        but is not the full long summary of the docstring.
    sec_header_re : re.Pattern
        Regex pattern for finding section headers within the docstring.
    param_header_re : re.Pattern
        Regex pattern for finding the "Parameter" or "Arguments" section header
        within the docstring.
    """
    def __init__(self, style, desc='paragraph'):
        """Constructs the parser's regex patterns based on the given style.

        Parameters
        ----------
        style : {'numpy', 'google', 'exputils'}
            The docstring style to be expected for parsing.
        desc : {'paragraph', 'short', 'long'}, optional
            The part of the docstring to include as the resulting argparse
            parser's description. Defaults to 'paragraph' to take the short
            summary from the docstring and any immediate text up to the next
            newline or section. 'short' only uses the short summary. 'long'
            uses the long summary.
        """
        style = style.lower()

        if style == 'numpy':
            sec_header = r'\n\s*{}\n\s*-+\n'
            # TODO handle `.. depracted::` too. But only matters if there is no
            # preceding newline given how `sec_header` is used in this parsing.
            # Incomplete parser cuz a full parser is not needed.
        elif style == 'google':
            raise NotImplementedError(' '.join([
                'Section headers in Google docstring style are same as params',
                'and thus rely on checking indentation. TODO.',
            ]))
            sec_header = r'\n\s*{}\n\s*-+\n'
        elif style == 'exputils':
            sec_header = r'\n\s*{}\n\s*-+\n'
            # TODO add default PEP8 docstring style
        else:
            raise ValueError(' '.join([
                f'Expected the styles "numpy", "google", or "exputils", but',
                f'recieved `{style}`.',
            ]))

        self.sec_header_re = re.compile(sec_header.format(r'\w+'))
        self.param_header_re = re.compile(
            sec_header.format(r'(Parameters|Args|Arguments)')
        )
        self.style = style

        if desc == 'paragraph':
            # Text from beginning to next section, empty line, or end if none.
            self.desc_re = re.compile(rf'.*($|{self.sec_header_re}|\n\s\n)')
        elif desc == 'short':
            # Get the text from beginning up to first newline or end if none.
            self.desc_re = re.compile(r'.*(\n|$)')
        elif desc == 'long':
            # Ends at new section
            self.desc_re = re.compile(rf'.*($|{self.sec_header_re})')
        else:
            raise ValueError(' '.join([
                f'Expected the desc to be of values "paragraph", "short", or',
                f'"long", but recieved `{desc}`.',
            ]))

        self.desc = desc

    def from_class(self):
        return

    def get_section_end_idx(doc_lines, start_idx=0, style='numpy'):
        """Find the index that is the final line of the section of the docstring
        given the initial index point to start from.

        Parameters
        ----------
        doc_lines : list of strings obtained from obj.__doc__.splitlines()
        start_idx : int, optional
        style : str, optional
        """
        if style == 'numpy':
            section_regex =
        elif style == 'google':
            raise NotImplementedError('Current regex includes params too.')
            section_regex = r'\n\s*\w+:\n'
        elif style == 'exputils':
            section_regex = r'\n\s*\[\w+\]\n'
        else:
            raise NotImplementedError(f'Unrecognized style: `{style}`')

        end_desc = 0
        length = len(doc_lines) - 1
        for i, line for enumerate(doc_lines, start_idx):
            end_desc = i
            stripped_line = line.strip()

            if '.. deprecated::' in stripped_line:
                # Obtain the short summary, i.e. that is the first paragraph.
                # Stop in case of a deprecation warning w/o empty line prior
                break
            elif set(stripped_line) == {'-'}:
                # Stop in case of no empty line prior to a section
                end_desc -= 1
                break
            elif i == length:
                # There is content in last line that is part of short summary.
                end_desc = length + 1

        return end_desc


    def parse_description(obj):
        """Parse the description from `obj.__doc__` expecting the given docstring
        style. The description is the short summary and assumes to be those whose
        signature is available, thus not written as the first line of the
        `__doc__`.
        """
        if obj.__doc__ is None:
            return ''

        #return self.desc_re.find

        doc_lines = obj.__doc__.splitlines()

        if len(doc_lines) > 1:
            return doc_lines[0].strip()

        end_desc = 0
        length = len(doc_lines) - 1
        for i, line for enumerate(doc_lines):
            end_desc = i
            stripped_line = line.strip()

            if stripped_line == '' or '.. deprecated::' in stripped_line:
                # Obtain the short summary, i.e. that is the first paragraph.
                # Stop in case of a deprecation warning w/o empty line prior
                break
            elif set(stripped_line) == {'-'}:
                # Stop in case of no empty line prior to a section
                end_desc -= 1
                break
            elif i == length:
                # There is content in last line that is part of short summary.
                end_desc = length + 1

            # Modify doc lines so it strips whitespace before joining
            doc_lines[i] = stripped_line

        return ' '.join(doc_lines[end_desc])


    def parse_params(obj, style='numpy'):
        """Parse the parameters from `obj.__doc__` epecting the given docstring
        style
        """
        if style not in {'numpy', 'exputils'}:
            # A personal modification to numpy styling
            raise NotImplementedError(' '.join([
                f'Only style able to parsed is "numpy". The `{style}` style is',
                'not yet supported.',
            ]))

        # TODO find Parameters section in docstring

        # TODO find end of Parameters section in docstring

        return


    def get_func_argparser(

    def from_func(
        self,
        parser,
        func,
        #style='numpy',
        exclude=None,
        abbrev=None,
        *args,
        **kwargs,
    ):
        """Get the argparse parser that corresponds to the docstring.

        Parameters
        ----------
        parser : argparse.ArgumentParser, optional
            The parser in which the resulting parser becomes an argument group to.
        obj : Object
            An object with a `__doc__` attribute that contians a docstring to be
            parsed.
        style : str | func
            The identifier of which docstring parser to use. Supported styles are
            'numpy, 'google', and 'pep8'. TODO add use of custom docstring parser
            by passing a callable object that when given a str parses that string
            and returns some standard format to be decided.
        exclude : list(str), optional
            A list of str identifiers that correspond to the arguments to be
            excluded from the argparse parser.
        abbrev : list(str) | dict(str:str) | bool, optional
            If a list or strings, then expects to abbreviate the arguments that are
            included in the parser by order of their position in the docstring. If
            a dictionary, then expects the keys to be the string names of the
            arguments that map to the string values of the abbreviations to be used
            for them. If True, then the abbreviations of each argument are created
            based on the first character of each argument name. When None, the
            default, no argument abbreviation occurs.

        Returns
        -------
        argparse.ArgumentParser | argparse._ArgumentGroup
            The parser resulting from the parsed docstring of the given object. If
            `parser` was given, then the ArgumentGroup object is returned.
        """
        if hasattr(func, '__doc__'):
            raise AttributeError(' '.join([
                f'The object `func` of type {type(func)} does not have a',
                '`__doc__` attribute. Cannot create parser for this object.',
            ]))

        parser_group = parser.add_argument_group(
            func.__name__,
            parse_description(func.__doc__),
        )

        parsed_params = parse_params(func, style)

        for param, param_vars in parsed_params.items():
            name = f'--{param}' if 'default' in param_vars else param

            if abbrev:
                add_argument = parser_group.add_argument
            else:
                add_argument = parser_group.add_argument

            add_argument(
                name,
                **param_vars,
            )

        # TODO allow for extracting arg data from other object's existing
        # docstrings. This is useful when the same arg is passed around
        # multiple times or expected to be the same in different places. This
        # would be accomplished by simply have some identifier that a docstring
        # hyperlink is occuring, e.g. `package.module.submodule.func`, maybe
        # with optional `...func(arg_name)` if different than current one's
        # name. This would sacrifice locality of that information to the code
        # for the sake of "write once" design. The docstring when rendered in
        # autodocs could then include the link to the docs being linked and
        # even expand the docs inplace to avoid making the user traverse links
        # more than necessary (solving locality issue in rendered docs, but not
        # code).

        return parser


def get_class_argparser(class_obj, methods, *args, **kwargs):
    """Creates the argparse parser that contains the ArgumentGroups for each
    method whose docstrings are parsed.

    Parameters
    ----------
    class_obj : type
    methods : list(str)

    Returns
    -------
    argparse.ArgumentParser | argparse._ArgumentGroup
        The parser resulting from the parsed docstring of the given object. If
        `parser` was given, then the ArgumentGroup object is returned.
    """
    if hasattr(class_obj, '__doc__'):
        raise AttributeError(' '.join([
            f'The object `class_obj` of type {type(class_obj)} does not have',
            'a `__doc__` attribute. Cannot create parser for this object.',
        ]))

    docstring = func.__doc__

    return parser


def get_argparser(obj, *args, **kwargs):
    """Convenience function that determines if the given object is a class or
    function and creates the appropriate argparse parser.
    """
    if callable(obj):
        # Treat as function
        get_func_argparser()
    elif isinstance(obj, type):
        # Treat as class
        get_class_argparser()
    else:
        raise TypeError(' '.join([
            'Expected a class or function, but recieved an object of type',
            f'`{type(obj)}`',
        ]))

    return


# TODO check primitives, confirm all primitives and basic numpy dtypes are as
# stated in docstring; do this as a decorator, separate from the parser
# creation. This is mostly for fun and experience, but if properly generalized,
# then it wwould be useful and simplify code when such checks need to be done
# prior to running anything.

# TODO Possibly could add an extra one that checks a given set of conditions if
# generic primitive and numpy dtype checking is not enough. Custom type
# checking by providing a function or multiple functions.
