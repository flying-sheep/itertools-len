import sys

sys.path.insert(0, "..")

project = "itertools-len"
copyright = "2020, Philipp A."
author = "Philipp A."

nitpicky = True

# HTML
html_theme = "sphinx_rtd_theme"
html_theme_options = dict(collapse_navigation=False)
html_context = dict(
    display_github=True,
    github_user="flying-sheep",
    github_repo="itertools-len",
    github_version="master",
    conf_py_path="/docs/",
)

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]
intersphinx_mapping = dict(python=("https://docs.python.org/3", None))
