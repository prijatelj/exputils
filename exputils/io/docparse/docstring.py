"""Code for managing docstring parsing and conversion."""

# Modify `sphinxcontrib.napoleon` to include `exputils` mod to Numpy style
# Use respective style: `sphinxcontrib.napoleon.GoogleDocstring(docstring,
# config)` to convert the docstring into ReStructuredText.
# Parse that restructured text to obtain the args, arg types, decsriptions,
# etc.

# TODO Docstring object that contains the parts of the docstring post parsing

class Docstring(object):
    """Docstring components and parsing."""

    def __init__(self, style)
