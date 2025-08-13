from setuptools import setup, find_packages

setup(
    name='brain-connectivity-graph',
    version='0.1.0',
    author='Bailey Ng',
    author_email='bailey.ng@mail.utoronto.ca',
    description='A project to convert functional connectivity matrices into graph representations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bailey168/MIND_models/brain-connectivity-graph',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'networkx',
        'pandas'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)