project = "spatialPeeler"
copyright = "2026, goeva lab"
author = "goeva lab"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conf.py", "demo/_shared.py"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "source repository",
            "url": "https://github.com/goeva-lab/spatialPeeler",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ]
}

html_css_files = ["custom.css"]

nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}
nb_execution_timeout = -1
nb_execution_show_tb = True
nb_code_prompt_show = "show"
nb_code_prompt_hide = "hide"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

typehints_use_rtype = False

myst_enable_extensions = ["alert"]
