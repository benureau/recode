#!/usr/bin/env python

import sys
import os
import time
import subprocess


this_dir = os.path.abspath(os.path.dirname(__file__))

def publish(filepath, timeout=1200):
    prefix, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    os.makedirs('published/{}'.format(prefix), exist_ok=True)
    if ext == '.ipynb':
        # .py file
        cmd = ('jupyter-nbconvert --to python '
               '--ClearOutputPreprocessor.enabled=True '
               '{} --output {}.py')
        subprocess.check_call(cmd.format(filepath, name).split())

        # published .py file
        cmd = ('jupyter-nbconvert --to python '
               '--ClearOutputPreprocessor.enabled=True '
               '--ExecutePreprocessor.timeout={} '
               '{} --output {}/published/{}.py')
        subprocess.check_call(cmd.format(timeout, filepath, this_dir, os.path.join(prefix, name)).split())

        # published .ipynb file
        start = time.time()
        cmd = ('jupyter-nbconvert --to notebook '
               '--ClearOutputPreprocessor.enabled=True '
               '--ExecutePreprocessor.timeout={} '
               '--execute '
               '{} --output {}/published/{}')
        subprocess.check_call(cmd.format(timeout, filepath, this_dir, filepath).split())
        print('took {:.2f} seconds'.format(time.time() - start))

        # published .html file
        cmd = ('jupyter-nbconvert --to html '
               'published/{} --output {}/published/{}.html')
        subprocess.check_call(cmd.format(filepath, this_dir, os.path.join(prefix, name)).split())


def publish_folder(path):
    for dirpath, dirnames, filenames in os.walk(path):
        if os.path.basename(dirpath) != '.ipynb_checkpoints':
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                publish(filepath)


if __name__ == '__main__':

    assert len(sys.argv) > 1
    filepaths = sys.argv[1:]

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print("error: {} not found".format(filepath))
            sys.exit(1)


    for filepath in filepaths:
        if os.path.isfile(filepath):
            publish(filepath)
        else:
            publish_folder(filepath)
