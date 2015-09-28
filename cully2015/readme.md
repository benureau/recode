# Recode: Robots that can adapts like animals

We recode the arm experiment of the article "Robot that can adapt like animals" ([DOI](http://dx.doi.org/10.1038/nature14422)) by Antoine Cully, Jeff Clune, Danesh Tarapore and Jean-Baptiste Mouret. The article is available [on the Nature website](http://www.nature.com/nature/journal/v521/n7553/full/nature14422.html), and a preprint [is available here](http://www.isir.upmc.fr/files/2015ACLI3468.pdf). The authors have made the [C++ code used for the experiments](http://pages.isir.upmc.fr/~mouret/code/ite_source_code.tar.gz) in the article available, but it was necessary to consult it to code this Python implementation. The [supplementary information](http://www.nature.com/nature/journal/v521/n7553/extref/nature14422-s1.pdf) document, however, was instrumental to it. This code is available on the [recode github repository](https://github.com/humm/recode), and is published under the [OpenScience License](http://fabien.benureau.com/openscience.html).

## Executing Online

The .ipynb file hosted on github does not contain the output (which is too large). The notebook can be [viewed and executed directly online](http://mybinder.org/repo/humm/recode/cully2015/cully2015.ipynb) without the need to install anything using the (still experimental) [binder service](http://mybinder.org/). An [html version with precomputed output](http://fabien.benureau.com/recode/cully2015.html) also is available. The jupyter notebook with output can also be viewed through [nbviewer](http://nbviewer.ipython.org/url/fabien.benureau.com/recode/cully2015.ipynb).

## Installing Locally

The code depends on the [numpy](http://www.numpy.org/), and the [bokeh](http://bokeh.pydata.org) library for the figures. In a terminal:
```
pip install -U numpy bokeh
```

In order to use the .ipynb files, [jupyter]() is required:
```
pip install -U jupyter
jupyter-notebook cully2015.ipynb
```
