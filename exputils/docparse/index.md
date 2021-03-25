## Docparse

Adjustments to the argument and config parsers in python with the intention to expedite the coding process, as well as make it easier to manage and extend.
Properly written docstrings contain the same information needed for creating an argparse or configuration parser, thus allowing these to be created from the parsed docstrings.
Some other common code writing tasks can also be automatically completed to reduce redundant code.

### Features

. . .

Relies on python dataclasses to contain the args and wraps parts of Sphinx for auto-documentation, (probably) ConfigArgparse for better configuration file support stand-in for argparse, and (possibly) pydantic for optional dataclass runtime type checking.
Turning the args/attributes sections of docstrings into extended dataclass code w/ following descriptoins.

#### TODO

- Nesting in argparse and configs
- Inheritance
- Ease of serialization of objects
- ArgConfig objects must be splat-able (\* and \*\*)
- Config loading and saving in argparse
    - Perhaps use ConfigArgParse as a backend to be modified, allowing nesting?
    - config support for YAML and JSON
- Metaprogramming for generating ArgConfig objects from python object's docstring.
    - basic function to generate the ArgConfig object for parsing and data passing
        - docstring parser and transformer (swap between styles) for
            - PEP8
            - Google
            - Numpy
            - exputils (custom)
                - Sphinx mod for parsing and generating docs
    - Inheritance / Class walking to obtain inheritance ArgConfig objects
    - write-once args expansion via a decorator
        - Parses the doc string given the style, and then wraps the function to expect the stated args and kwargs.
            - This parse will be costly, and thus needs to be computed once only.
                TF lib and others have done this says Tim, so I have to figure out how they do it.
                Apparently, the decorator itself can handle this w/ a stateful check.
        - optional type checking based on the docstring
            - create pylint / flake8 with some extension to either ignore checking such functions, or do the parse themselves and then lint.
