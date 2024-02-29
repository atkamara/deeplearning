from setuptools import setup,find_packages

description = open("readme.md").read()

setup(
    name="neural_net_numpy",
    version="0.1.1",
    author="Abdourahmane Tintou KAMARA",
    author_email="abdourahmane29@outlook.com",
    packages=find_packages(include=["neural_net"]),
    tests_require=["pytest>=4.4.1"],
    test_suite="tests",
    long_description=description, 
    long_description_content_type="text/markdown",
    url="https://github.com/atkamara/deeplearning/tree/main/Neural%20Net%20v0.1.0", 
    license="MIT", 
    python_requires=">=3.8", 
    install_requires=["pytest-runner","tqdm>=4.66.2","numpy>=1.26.4","SQLAlchemy>=2.0.27","pandas>=2.2.1"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: POSIX :: Linux",        
        "Programming Language :: Python :: 3",
    ]
)