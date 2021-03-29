import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='saphyra',
    version='3.1.1',
    license='GPL-3.0',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    author='João Victor da Fonseca Pinto',
    author_email='jodafons@lps.ufrj.br',
    url='https://github.com/ringer-atlas/saphyra',
    keywords=['framework', 'flexibility', 'python',
              'online', 'machine learning', 'deep learning'],
    install_requires=[
        # 'tensorflow', # this cause an missmatch into the tensorflow docker image. We will install manually in case of not found.
        "Gaugi>=1.0.13",
        "keras2onnx>=1.7.0",
        "matplotlib>=3.4.0",
        "numpy>=1.18.1",
        "onnx>=1.8.1",
        "pandas>=1.2.3",
        "scikit_learn>=0.24.1",
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
