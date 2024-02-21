from setuptools import find_packages, setup

setup(
    name="starling",
    version="0.1.0",
    url="https://github.com/camlab-bioml/starling",
    project_urls={
        "Issues": "https://github.com/camlab-bioml/starling/issues",
        "Source": "https://github.com/camlab-bioml/starling",
    },
    author="Jett (Yuju) Lee",
    author_email="yulee@lunenfeld.ca",
    packages=find_packages(),
    package_dir={"starling": "starling"},
    package_data={"": ["*.json", "*.html", "*.css"]},
    include_package_data=True,
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["imaging cytometry classifier single-cell"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
    ],
    license="See License.txt",
    install_requires=[
        "pip==23.2.1" "torch==1.12.1",
        "tensorflow==2.14.0",
        "pytorch-lightning==2.1.0",
        "scanpy==1.9.5",
    ],
    python_requires=">=3.9.0",
)
