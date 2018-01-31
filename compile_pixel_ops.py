#C:\Users\Neil\Anaconda\python.exe compile_cython.py build_ext --inplace  --compiler=msvc
from setuptools import setup
from setuptools import Extension
#from distutils.core import setup
#from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("pixel_ops",
                         ["pixel_ops.pyx"],
                         #extra_compile_args=['-fopenmp'],
                         #extra_link_args=['-fopenmp'],
                         )]
setup(
    name = "Pixel level operations",
    cmdclass = {"build_ext": build_ext},
    include_dirs = [np.get_include()],
    ext_modules = ext_modules
)

print "hi"