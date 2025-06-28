from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fdwm",
    version="0.2.2",
    author="Liam Huang",
    author_email="PyPI@liam.page",
    description="Frequency-domain watermarking library and CLI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Liam0205/fdwm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "fdwm=fdwm.cli:main",
        ],
    },
    include_package_data=True,
    keywords="watermark, steganography, image processing, frequency domain, fft",
    project_urls={
        "Bug Reports": "https://github.com/Liam0205/fdwm/issues",
        "Source": "https://github.com/Liam0205/fdwm",
        "Documentation": "https://github.com/Liam0205/fdwm#readme",
    },
)
