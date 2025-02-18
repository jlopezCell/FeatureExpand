from setuptools import setup, find_packages

setup(
    name='featureexpand',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'requests',
        'json',
        'typing'
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'featureexpand=featureexpand.feature_expander:main',
        ],
    },
    author='Juan Carlos Lopez Gonzalez',
    author_email='jlopez1967@gmail.com',
    description='FeatureExpand is a powerful Python library designed to enhance your datasets by processing and generating additional columns. Whether you''re working on machine learning, data analysis, or any other data-driven application, FeatureExpand helps you extract maximum value from your data. With intuitive functions and easy extensibility, you can quickly add new features to improve the quality and metrics of your analysis and modeling.',
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
