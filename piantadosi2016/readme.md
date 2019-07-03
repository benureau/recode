# Recode: Extraordinary intelligence and the care of infants

We recode the model of the article "Extraordinary intelligence and the care of infants" ([10.1073/pnas.1506752113](https://www.pnas.org/cgi/doi/10.1073/pnas.1506752113)) by [Steve Piantadosi](http://colala.berkeley.edu/people/piantadosi/) and [Celeste Kidd](https://www.kiddlab.com). The pdf is [available here](https://www.celestekidd.com/papers/PiantadosiKidd2016Extraord.pdf). Here, we only succinctly describe the model. You should consult the original article for details and for the rationale behind the model's choices.

The spirit of this notebook is to use simple code that is easy to understand and modify. This notebook requires no specific knowledge beyond a basic grasp of the Python language. We show *all the code* of the model, without relying on any library beyond [numpy](https://www.numpy.org/). Only the plotting, using the [bokeh](https://bokeh.pydata.org/) library, have been abstracted away in the [graphs.py](https://github.com/humm/recode/blob/master/piantadosi2016/graphs.py) file. We employ the [reproducible](https://github.com/oist-cnru/reproducible) library to keep track of the computational environment and foster reproducibility.

Beside the raw notebook, you can access:

* a [html version](http://fabien.benureau.com/recode/piantadosi2016/piantadosi2016.html), with precomputed outputs.
* an [online interactive version](https://mybinder.org/v2/gh/benureau/recode/master?filepath=piantadosi2016%2Fpiantadosi2016.ipynb), powered by the [binder service](http://mybinder.org). This is still experimental; the computations may take a long time.
* a citable version of this notebook is available at [figshare](https://dx.doi.org/10.6084/m9.figshare.3990486).

You can contact me for questions or remarks at fabien@benureau.com.


## Installing Locally

The code depends on the [numpy](http://www.numpy.org/), and the [bokeh](http://bokeh.pydata.org) plotting library for the figures, the [and you will need [jupyter](http://jupyter.org/) to open the .ipynb file. In a terminal:

```
git clone http://github.com/benureau/recode
cd recode/piantadosi2016
pip install -r requirements.txt
jupyter-notebook piantadosi2016.ipynb
```
