#!/usr/bin/env python

# A small script to automatize creating the archive
import sys
import os
import time
import subprocess


here = os.path.abspath(os.path.dirname(__file__))

# clearing the notebook of outputs
notebook_filepath = os.path.join(here, 'piantadosi2016.ipynb')
cmd = ('jupyter-nbconvert', '--inplace', '--to', 'notebook',
       '--ClearOutputPreprocessor.enabled=True', notebook_filepath,
       '--output', notebook_filepath)
subprocess.check_call(cmd)

# producing the python and markdown version
cmd = ('jupyter-nbconvert', '--to', 'markdown', notebook_filepath)
subprocess.check_call(cmd)
cmd = ('jupyter-nbconvert', '--to', 'script', notebook_filepath)
subprocess.check_call(cmd)
