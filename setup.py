from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="petroleum-to-planet",
    version="1.0.0",
    author="Zahara",
    description="Environmental Data Science Portfolio: From Petroleum Engineering to Planet Protection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/petroleum-to-planet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Environmental Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "xarray>=2023.6.0",
        "scipy>=1.11.1",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "cartopy>=0.21.1",
        "plotly>=5.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.1",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "env-pipeline=run_pipeline:main",
        ],
    },
)
