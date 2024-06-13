# conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Inspect AI Scorers'
copyright = '2023, Abigail Haddad'
author = 'Abigail Haddad'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']