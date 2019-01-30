from distutils.core import setup, Extension

module1 = Extension('pytoep',
        sources = ['pytest.c', 'toeplitz.c'])

setup (name = 'pytoep',
        version = '1.0',
        description = 'This is a demo package',
        ext_modules = [module1])
