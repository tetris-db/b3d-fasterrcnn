"""Build the Cython OC-SORT tracker in place: ./run setup.py build_ext --inplace

The tracker (ocsort/) is copied from the Tetris repository (polyis/tracker/ocsort/cython, commit
8404d4f), and lapjv (ocsort/lapjv, BSD-2-Clause) from github.com/chanwutk/lap at 58eee4d, which
Tetris vendors as modules/lap. Compiler flags match Tetris's setup.py.
"""
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


# Tetris's optimization flags for its tracker extensions.
ARGS = ['-O3', '-ffast-math', '-march=native', '-mtune=native', '-finline-functions']
# Tetris targets NumPy 2.3's API; this image ships NumPy 1.21, so use the 1.7 API (no deprecated calls).
MACROS = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

extensions = [
    Extension('ocsort.kalman_filter', ['ocsort/kalman_filter.pyx'],
              include_dirs=[numpy.get_include()], define_macros=MACROS, extra_compile_args=ARGS),
    Extension('ocsort.association', ['ocsort/association.pyx', 'ocsort/lapjv/lapjv.cpp'],
              include_dirs=['ocsort/lapjv', numpy.get_include()], define_macros=MACROS,
              extra_compile_args=ARGS + ['-std=c++11'], language='c++'),
    Extension('ocsort.ocsort', ['ocsort/ocsort.pyx'],
              include_dirs=[numpy.get_include()], define_macros=MACROS,
              extra_compile_args=ARGS + ['-std=c++11'], language='c++'),
]

# Tetris's compiler directives, except freethreading_compatible and subinterpreters_compatible,
# which need Cython 3.1 (this image's Python 3.7 caps Cython at 3.0) and only affect threading modes not used here.
DIRECTIVES = {
    'language_level': 3, 'boundscheck': False, 'wraparound': False, 'initializedcheck': False,
    'nonecheck': False, 'overflowcheck': False, 'overflowcheck.fold': False, 'embedsignature': False,
    'cdivision': True, 'cpow': True, 'optimize.use_switch': True, 'optimize.unpack_method_calls': True,
    'infer_types': True, 'profile': False, 'linetrace': False, 'emit_code_comments': False,
    'annotation_typing': False, 'c_string_type': 'str', 'c_string_encoding': 'ascii',
    'type_version_tag': True, 'unraisable_tracebacks': False, 'iterable_coroutine': True, 'fast_gil': True,
}

setup(name='ocsort', ext_modules=cythonize(extensions, compiler_directives=DIRECTIVES))
