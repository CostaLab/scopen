import io
import os
import re

# from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("cdnmf_fast", ["./scopen/cdnmf_fast.pyx"]),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("cdnmf_fast", ["./scopen/cdnmf_fast.c"]),
    ]


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


current_version = find_version("scopen", "__version__.py")

setup(
    name='scOpen',
    version=current_version,
    packages=['scopen'],
    author='Zhijian Li',
    author_email='zhijian.li@rwth-aachen.de',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    entry_points={
        'console_scripts': [
            'scopen = scopen.Main:main'
        ]},
    install_requires=['numpy',
                      'h5py',
                      'six',
                      'pandas',
                      'scipy',
                      'tables',
                      'matplotlib',
                      'scikit-learn',
                      'kneed']
)
