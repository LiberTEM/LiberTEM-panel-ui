from setuptools import setup, find_packages

setup(
    name="udfs_ui",
    version='0.0.1',
    license='MIT',
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.7.5',
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-image",
        "tifffile",
        "PyYAML",
        "xarray",
        "bokeh>=2.4.2",
        "panel",
        "datashader",
        "colorcet",
        "typing_extensions",
        "notebook",
        "ipywidgets",
        "ipython",
        "libertem @ git+https://github.com/LiberTEM/LiberTEM.git@master",
        "libertem-live",
        "humanize",
        "bidict",
        "xarray",
        "datashader",
    ],
    extra_requires={
        "lt": [
            "libertem",
        ],
        "cp": [
            "cupy",
            "cupyx",
        ],
        'notebook': [
            "notebook",
            "ipywidgets",
            "ipython",
        ]
    },
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    entry_points={},
    description="Panel+Bokeh-based GUI Toolkit for image data processing",
    long_description='''
Provides a web-based, modular UI toolkit for image data processing.
Supports both single-page web-app workflows and running indie Jupyter
notebooks, allowing advanced users to add UI tools to their analyses
or create step-by-step analysis pipelines for novice users.

Created originally to support the LiberTEM electron microscopy
data analysis library (https://github.com/LiberTEM/LiberTEM).
''',
    url="https://github.com/matbryan52/aperture",
    author_email="libertem-dev@googlegroups.com",
    author="Matthew Bryan",
    keywords="electron microscopy, web ui, image processing",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Environment :: Web Environment',
        'Environment :: Console',
    ],
)
