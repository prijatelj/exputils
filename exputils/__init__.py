from importlib import import_module
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
    globals()[module] = import_module(f'.{module}', 'exputils')
