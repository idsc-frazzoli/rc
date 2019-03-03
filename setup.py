from setuptools import setup, find_packages

setup(name='carma',
      package_dir={'': 'src'},
      packages=find_packages('src'),

      zip_safe=False,
      entry_points={
            'console_scripts': [
                  'carma1 = carma:carma1_main',
                  'carma2 = carma:carma2_main',
            ]
      },
      install_requires=[
            'pybase64',
            'PyContracts',
            'IPython',
            'validate_email',
            'mypy_extensions',
            'nose',
            'coverage',
            'networkx',
            'dataclasses',
            'jsonschema',
            'pydot',
        ],
      )
