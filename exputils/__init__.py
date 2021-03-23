__version__ = '0.2.0'

__all__ = [
    'io',
    'data',
    'docparse',
    'ml',
    'profile',
    'ray',
    'visuals',
]

for module in __all__:
    exec(f'from exputils import {module}')
