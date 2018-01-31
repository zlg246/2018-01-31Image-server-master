#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("cc_label",
                         ["cc_label.pyx"],
                         )]
setup(
    name = "Connected component labelling",
    cmdclass = {"build_ext": build_ext},
    include_dirs = [np.get_include()],
    ext_modules = ext_modules
)
