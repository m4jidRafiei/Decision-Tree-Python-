import setuptools

import p_decision_tree

def read_file(filename):
    with open(join(dirname(__file__), filename)) as f:
        return f.read()


setuptools.setup(
    name=p_decision_tree.__name__,
    version=p_decision_tree.__version__,
    author=p_decision_tree.__author__,
    author_email=p_decision_tree.__author_email__,
    description="Visual Decision Tree Based on Categorical Attributes Package",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/m4jidRafiei/Decision-Tree-Python-",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'graphviz==0.9',
        'pandas==0.24.2'
    ],
    project_urls={
        'Source': 'https://github.com/m4jidRafiei/Decision-Tree-Python-'
    }
)