import setuptools

import p_decision_tree

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='p_decision_tree',
    version='0.0.3',
    author=p_decision_tree.__author__,
    author_email='majid.rafiei@pads.rwth-aachen.de',
    description="Visual Decision Tree Based on Categorical Attributes Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m4jidRafiei/Decision-Tree-Python-",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'graphviz',
        'pandas'
    ],
    project_urls={
        'Source': 'https://github.com/m4jidRafiei/Decision-Tree-Python-'
    }
)