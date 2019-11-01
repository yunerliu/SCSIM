import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="scsim", 
    version="0.0.1",
    author="Yunchen Liu, Hongming Pu, Yibo Yang",
    author_email="liuyunchen@pku.edu.cn",
    description="VAE based single-cell data simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)