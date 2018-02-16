import sys
import platform
import datetime
import subprocess

import pip


def provenance_statement(packages=(), provdata=None, savename='provenance'):
    if provdata is None:
        provdata = provenance(dirty_allowed=False)

    s = 'The code was executed with:\n    {} {}\n    on {}\n    at {}.\n'.format(
            provdata['python']['implementation'],
            '.'.join(provdata['python']['version']),
            provdata['platform'],
            '{} at {}'.format(*provdata['timestamp'].split('T')))

    if provdata['git']['dirty']:
        s+= ('git commit: {} (*dirty* repository). '
             'See full provenance data for the diff.\n'
            ).format(str(provdata['git']['revision']))
    else:
        s+= ('git commit: {} (fully-commited repository).\n'
            ).format(str(provdata['git']['revision']))

    if len(packages) > 0:
        s+= 'Partial list of packages:\n'
        s+= '\n'.join('    {}: {}'.format(pkg_name, provdata['packages'][pkg_name])
                for pkg_name in packages)
        s+= '\n'

    filename = '{}.txt'.format(savename)
    with open(filename, 'w') as f:
        f.write(str(provdata))

    s += 'The full provenance data has been saved as `{}.txt`.\n'.format(savename)
    return s


def provenance(dirty_allowed=False):
    """Return provenance data about the execution environment.

    Args:
        dirty_allowed:  if False, will exit with an error if the git repository
                        is dirty or absent.
    """
    return {'python'   : {'implementation': platform.python_implementation(),
                                'version' : platform.python_version_tuple(),
                                'compiler': platform.python_compiler(),
                                'branch'  : platform.python_branch(),
                                'revision': platform.python_revision()},
            'platform'  : platform.platform(),
            'packages'  : all_packages(), # list of installed packages
            'git'       : git_info(dirty_allowed=dirty_allowed),
            'timestamp' : datetime.datetime.utcnow().isoformat()+'Z',  # Z stands for UTC
           }

def all_packages():
    pkgs = {}
    for pkg_str in pip.commands.freeze.freeze():
        pkg_name, pkg_version = pkg_str.split('==')
        pkgs[pkg_name] = pkg_version
    return pkgs




class GitDirty(Exception):
    pass

def git_info(dirty_allowed=False):
    """Try to retrieve the git revision and dirty state.

    Return:  a dict with the retrieved values.

    Args:
        dirty_allowed  If True, and the repo is dirty, a diff will be included
                       in the returned values. If False and the repo is dirty,
                       a GitDirty exception is raised.
    """
    # find out if we are dirty...
    try:
        status_cmd = ('git', 'status', '--porcelain')
        dirty = len(subprocess.check_output(status_cmd)) > 0
    except subprocess.CalledProcessError: # non-zero status: repo probably not found
        if dirty_allowed:
            return {'error': 'git repository not found'}
        else:
            print('error: git repository not found', out=sys.err)
            raise GitDirty

    if dirty and not dirty_allowed:
        print('error: git repository is dirty; commit changes and retry', out=sys.err)
        raise GitDirty

    hash_cmd = ('git', 'rev-parse', 'HEAD')
    revision = subprocess.check_output(hash_cmd).decode('utf-8')[:-1]
    version_cmd = ('git', '--version')
    version = subprocess.check_output(version_cmd).decode('utf-8')

    if dirty:
        diff_cmd = ('git', 'diff')
        diff = subprocess.check_output(diff_cmd)
        return {'revision': revision, 'dirty': dirty, 'diff': diff}
    else:
        return {'revision': revision, 'dirty': dirty}
