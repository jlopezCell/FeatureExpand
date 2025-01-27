from setuptools import setup, find_packages

setup(
    name='featureexpand',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'featureexpand=featureexpand.feature_expander:main',
        ],
    },
    author='Juan Carlos Lopez Gonzalez',
    author_email='jlopez1967@gmail.com',
    description='A library to process and generate additional columns in any dataset',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jlopezCell/FeatureExpand',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
