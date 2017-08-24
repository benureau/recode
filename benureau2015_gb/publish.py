#!/usr/bin/env python

# A small script to automatize creating the archive
import sys
import os
import time
import subprocess


here = os.path.abspath(os.path.dirname(__file__))

# clearing the notebook of outputs
notebook_filepath = os.path.join(here, 'benureau2015_gb.ipynb')
cmd = ('jupyter-nbconvert', '--inplace', '--to', 'notebook',
       '--ClearOutputPreprocessor.enabled=True', notebook_filepath,
       '--output', notebook_filepath)
subprocess.check_call(cmd)

# producing the python and markdown version
cmd = ('jupyter-nbconvert', '--to', 'markdown', notebook_filepath)
subprocess.check_call(cmd)
cmd = ('jupyter-nbconvert', '--to', 'script', notebook_filepath)
subprocess.check_call(cmd)

# executing the notebook
output_notebook = os.path.join(here, 'benureau2015_gb_output.ipynb')
cmd = ('jupyter-nbconvert', '--to', 'notebook', '--ExecutePreprocessor.timeout=600',
       '--execute', notebook_filepath, '--output', output_notebook)
start = time.time()
subprocess.check_call(cmd)
print("Execution took {:.2f} seconds.".format(time.time() - start))

# exporting as html
cmd = ('jupyter-nbconvert', '--to', 'html', output_notebook)
subprocess.check_call(cmd)

# create the archive
cmd = ('tar', 'cfJ', 'benureau2015_gb.tar.xz',
                     'benureau2015_gb.ipynb',
                     'benureau2015_gb.py',
                     'benureau2015_gb.md',
                     'benureau2015_gb_output.ipynb',
                     'benureau2015_gb_output.html',
                     'readme.md',
                     'graphs.py',
                     'requirements.txt',
                     'publish.py')
subprocess.check_call(cmd)
