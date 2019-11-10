from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='scOpen',
    version='0.1',
    packages=['scopen'],
    ext_modules=cythonize("./scopen/cdnmf_fast.pyx"),
    entry_points={
        'console_scripts': [
            'scopen = scopen.__main__:main'
        ]},
    install_requires=['numpy',
                      'h5py',
                      'six>=1.10.0',
                      'pandas',
                      'scipy',
                      'tables',
                      'matplotlib']
)
