from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="covidsearch",
    version="0.0.2",
    author="Congcong Wang",
    author_email="wangcongcongcc@gmail.com",
    description="A customizable platform searching covid-19 relevant papers based on ",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/UKPLab/sentence-transformers",
    download_url="https://github.com/UKPLab/sentence-transformers/archive/v0.2.5.zip",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "tqdm",
        "flair",
        "gensim",
        "pandas",
        "flask",
        "flask_cors",
        "numpy",
        "scikit-learn",
        "scipy",
        "nltk"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="COVID-19, SearchAsAservice"
)
