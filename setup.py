from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="amq",
    version="0.1.0",
    author="Sangjun Lee, Seung-taek Woo",
    author_email="sangjunlee@postech.ac.kr, wst9909@postech.ac.kr",
    description="Automated Mixed-precision Quantization for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dlwns147/amq",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)
