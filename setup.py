import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().strip().split("\n")

setuptools.setup(
    name="flutes",
    version="0.3.0",
    url="https://github.com/huzecong/flutes",

    author="Zecong Hu",
    author_email="huzecong@gmail.com",

    description="Utilities to charm your Python",
    long_description=long_description,
    long_description_content_type="text/markdown",

    license="MIT License",

    packages=setuptools.find_packages(),
    package_data={
        "flutes": [
            "py.typed",  # indicating type-checked package
        ],
    },
    platforms='any',

    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Utilities",
        "Typing :: Typed",
    ],
    python_requires='>=3.6',
)
