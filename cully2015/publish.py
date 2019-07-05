#!/usr/bin/env python

# A small script to automatize creating the archive
import os
import argparse
import subprocess



def prepublish(notebook_filepath):
    """Stabilizing the git repository"""
    # clearing the notebook of outputs
    cmd = ('jupyter-nbconvert', '--inplace', '--to', 'notebook',
           '--ClearOutputPreprocessor.enabled=True', notebook_filepath,
           '--output', notebook_filepath)
    subprocess.check_call(cmd)

    # producing the python and markdown version
    cmd = ('jupyter-nbconvert', '--to', 'markdown', notebook_filepath)
    subprocess.check_call(cmd)
    cmd = ('jupyter-nbconvert', '--to', 'script', notebook_filepath)
    subprocess.check_call(cmd)

def publish(notebook_filepath):
    cwd_ = os.getcwd()
    dirname, notebook_filename = os.path.split(notebook_filepath)
    os.chdir(dirname)

    name, ext = os.path.splitext(notebook_filename)
    output_filename = name + '_output' + ext

    # executing the notebook
    cmd = ('jupyter-nbconvert', '--to', 'notebook', '--ExecutePreprocessor.timeout=600',
           '--execute', notebook_filename, '--output', output_filename)
    subprocess.check_call(cmd)

    # exporting as html
    cmd = ('jupyter-nbconvert', '--to', 'html', output_filename)
    subprocess.check_call(cmd)

    # create the archive
    cmd = ('tar', 'cfJ', '{}.tar.xz'.format(name),
                         notebook_filename,
                         '{}.py'.format(name),
                         '{}.md'.format(name),
                         output_filename,
                        '{}_output.html'.format(name),
                        'readme.md',
                        'graphs.py',
                        'requirements.txt',
                        'publish.py')
    print(cmd)
    subprocess.check_call(cmd)

if __name__ == '__main__':
    here = os.path.abspath(os.path.dirname(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--pre', help="run only prepublish routines",
                        action='store_true', default=False)
    args = parser.parse_args()

    notebook_filepath = os.path.join(here, 'cully2015.ipynb')
    prepublish(notebook_filepath)
    if not args.pre:
        publish(notebook_filepath)
