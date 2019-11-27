from distutils.core import setup, Extension

e = Extension('_sample', ['mctimme2/c/_samplemodule.c'], include_dirs=[])
e2 = Extension('distribution', ['mctimme2/c/distributionmodule.c'], include_dirs=[])

extensions = [e, e2]

setup(name='MCTIMME2_model',
      install_requires = ['pandas',
                          'ujson',
                          'numpy',
                          'matplotlib',
                          'scipy',
                          'sklearn', 'h5py', 'numba'],
      include_package_data = True,
      packages = ['mctimme2',],
      ext_modules = extensions,
      entry_points = {'console_scripts': ['MCTIMME2_model=mctimme2.run:run_model']},
      )
