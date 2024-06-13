from setuptools import setup, find_namespace_packages

# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Remove any comments and empty lines
    requirements = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name='inspect_ai_scorers',
    version='0.1.6',
    description='Adds scorers for usage with the inspect_ai package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Abigail Haddad',
    author_email='abigail.haddad@gmail.com',
    url='https://github.com/abigailhaddad/inspect_ai_eval',
    license='MIT',
    packages=find_namespace_packages(include=['inspect_ai_scorers', 'tests', 'examples']),
    package_data={
        'inspect_ai_scorers': ['logs/*.json'],
        'tests': ['logs/*.json'],
        'examples': ['*'],
    },
    exclude_package_data={
        '': ['__pycache__'],
    },
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Testing'
    ],
    python_requires='>=3.10',
)