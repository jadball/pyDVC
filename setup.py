import setuptools

# read README for long description
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name='pyDVC',
    version='0.0.3',
    packages=setuptools.find_packages(),
    url='https://github.com/jadball/pyDVC',
    license='BSD 3-Clause',
    author='James Ball',
    author_email='jadball@gmail.com',
    description='Python implementation of Discrete Voronoi Chain algorithm by Mirko Velić, Dave May & Louis Moresi (2009).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    py_modules=["pyDVC"],
    python_requires='>=3.6',
    # package_dir={'':'./'},
    install_requires=['numpy', 'numba']
)
