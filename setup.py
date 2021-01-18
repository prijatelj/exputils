from setuptools import setup

long_description = ''
with open('README.md', 'r') as f:
    long_description = f.read()

install_requires = ''
with open('requirements.txt', 'r') as f:
    install_requires = f.read()

setup(
    name='exputils',
    version='0.1.2',
    author='Derek S. Prijatelj',
    author_email='dprijate@nd.edu',
    packages=[
        'exputils',
        'exputils.data',
    ],
    #scripts
    description='Convenient functions that are commonly used for running machine learning experiments.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/prijatelj/exputils',
    install_requires=install_requires,
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
