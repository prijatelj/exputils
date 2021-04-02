from setuptools import setup
import re

def get_property(prop, project):
    """Gets the given property by name in the project's first init file."""
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + '/__init__.py').read()
    )
    return result.group(1)

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read()

project_name = 'exputils'

setup(
    name=project_name,
    version=get_property('__version__', project_name),
    author='Derek S. Prijatelj',
    author_email='dprijate@nd.edu',
    packages=[
        project_name,
        f'{project_name}.data',
        #f'{project_name}.io',
        f'{project_name}.ml',
        f'{project_name}.ray',
        f'{project_name}.visuals',
    ],
    #scripts
    description=' '.join([
        'Convenient functions that are commonly used for running machine',
        'learning experiments.',
    ]),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/prijatelj/{project_name}',
    install_requires=install_requires,
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
