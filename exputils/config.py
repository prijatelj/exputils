"""Create argparse compatible configuration objects of objects in Python as
determined by their docstring.
"""
import argparse

# TODO create the function when given any function or class, takes the
# docstring, optionally checks if docstring roughly matches class/args to avoid
# issues mismatching docstring to object, and optionally creates the argparse
# parser for that object using the information from the docstring.
def parse_description(obj, style='numpy'):
    """Parse the description from `obj.__doc__` expecting the given doc string
    style. The description is the short summary and assumes to be those whose
    signature is available, thus not written as the first line of the
    `__doc__`.
    """
    if style != 'numpy':
        raise NotImplementedError(' '.join([
            f'Only style able to parsed is "numpy". The `{style}` style is',
            'not yet supported.',
        ]))

    if obj.__doc__ is None:
        return ''

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
    """Parse the parameters from `obj.__doc__` epecting the given doc string
    style
    """

    return


def get_func_argparser(
    parser,
    func,
    style='numpy',
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

    return parser


def get_class_argparser(class_obj, methods, *args, **kwargs):
    """Creates the argparse parser that contains the ArgumentGroups for each
    method whose doc strings are parsed.

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


def get_arg_parser(obj, *args, **kwargs):
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
# generic primitive and numpy dtype checking is not enough.
