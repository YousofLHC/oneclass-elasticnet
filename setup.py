from setuptools import setup, find_packages

setup(
    name="oneclass-elasticnet",
    version="0.0.1",
    description="A package for anomaly detection using Elastic Net and Convex Hull.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # Content type for PyPI
    author="Yousof Ghalenoei", 
    author_email="yousof.ghalenoei2017@gmail.com",  
    url="https://github.com/YousofLHC/flexi_datareader",  # URL of your project
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        "scikit-learn>=1.6.1", 
        "numpy>=1.26.4",
        "qpsolvers>=4.3.",
        "tqdm>=4.66.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires=">=3.7",  # Minimum required Python version
)
