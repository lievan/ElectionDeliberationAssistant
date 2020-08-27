from setuptools import setup, find_packages

setup(
    name='ElectionDeliberationAssistant',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'gensim',
        'nltk',
        'transformers',
        'torch',
        'numpy',
    ],
    url='',
    license='Attribution-NonCommercial-ShareAlike 3.0',
    author='Evan Li',
    author_email='el3078@columbia.edu',
    description='Returns texts mined from Wikipedia on Trump and Biden relevant to a given query '
)