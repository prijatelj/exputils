from importlib import import_module
__version__ = '0.1.7rc0'

__all__ = [
    'io',
    'data',
    'ml',
    'profile',
    #'ray',
    #'visuals',
]

for module in __all__:
    globals()[module] = import_module(f'.{module}', 'exputils')
del import_module, module
