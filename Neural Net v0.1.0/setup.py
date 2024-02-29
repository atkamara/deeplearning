from setuptools import setup

with open("README.md", "r") as fh: 
    description = fh.read() 

setup(
    name="Neural Net",
    version="0.1.0",
    author="Abdourahmane Tintou KAMARA",
    author_email="abdourahmane29@outlook.com",
    packages=["src"],
    long_description=description, 
    long_description_content_type="text/markdown",
    url="https://github.com/atkamara/deeplearning/Neural Net v0.1.0", 
    license='MIT', 
    python_requires='>=3.8', 
    install_requires=["tqdm>=4.66.2","numpy>=1.26.4","SQLAlchemy>=2.0.27","pandas>=2.2.1"]
)