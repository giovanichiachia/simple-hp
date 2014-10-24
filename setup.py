from setuptools import setup, find_packages

setup(
    name='simplehp',
    version='0.1dev',
    packages=find_packages(),
    description='A simplifying wrapper of hyperopt and hyperopt-convnet.',
    license='BSD 3-clause license',
    long_description=open('README.md').read(),
    install_requires=['numpy',
                      'scipy',
                      'joblib',
                      'hyperopt',
                      'hpconvnet',
                      'scikit-learn'],
    package_data={'': ['*.md']},
)
