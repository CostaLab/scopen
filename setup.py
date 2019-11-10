import io
import os
import re

from distutils.core import setup
from Cython.Build import cythonize


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
    ext_modules=cythonize(["./scopen/cdnmf_fast.pyx"]),
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
