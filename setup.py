from setuptools import setup, find_packages

setup(
    name="analytics_store",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "polars>=0.20.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.0.0"
    ],
    author="Wicks Analytics LTD",
    description="A Python package for data analysis and analytics using Polars",
    python_requires=">=3.8",
    license="MIT",
    url="https://github.com/wicks-analytics/analytics_store",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
