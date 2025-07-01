from setuptools import setup, find_packages

setup(
    name='topogat',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'networkx',
        'gudhi',         # or ripser for PH
        'tqdm'
    ],
    author='Your Name',
    description='TopoGAT: Topological Attention for Graph Representation Learning using Persistent Homology',
    license='Apache 2.0',
)
