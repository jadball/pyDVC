import setuptools

setuptools.setup(
    name='pyDVC',
    version='1.0',
    packages=setuptools.find_packages(),
    url='https://github.com/jadball/pyDVC',
    license='BSD 3-Clause',
    author='James Ball',
    author_email='jadball@gmail.com',
    description='Python implementation of Discrete Voronoi Chain algorithm by Mirko Velić, Dave May & Louis Moresi (2009).',
    long_description=
    """Python implementation of Discrete Voronoi Chain algorithm.
    First described by Velić, Mirko & May, Dave & Moresi, Louis. (2009). A Fast Robust Algorithm for Computing Discrete Voronoi Diagrams. J Math Model Algor. 8. 343-355. 10.1007/s10852-008-9097-6.
    http://dx.doi.org/10.1007/s10852-008-9097-6""",
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
