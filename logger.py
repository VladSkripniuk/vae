import sys
import os
import shutil

import tempfile

from traceback import print_exception, format_exception

import datetime
import time

import _pickle

import numpy

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def is_picklable(obj):
  try:
    _pickle.dumps(obj)
  except pickle.PicklingError:
    return False
  return True

class Logger():
    def __init__(self, log_dirname, script_filename):

        self.log_dirname = log_dirname
        self.script_filename = script_filename

        if log_dirname is None or not os.path.isdir(log_dirname):
            self.log_dirname = "LOG_{}_{}".format(script_filename, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S'))

            os.makedirs(self.log_dirname)

        shutil.copy(self.script_filename, self.log_dirname)

        self.tmp_dirname = tempfile.mkdtemp()
        # print(self.tmp_dirname)
        self.f_stdout = open(os.path.join(self.tmp_dirname, 'stdout'), 'w+')
        self.f_stderr = open(os.path.join(self.tmp_dirname, 'stderr'), 'w+')

        sys.stdout = self.f_stdout
        sys.stderr = self.f_stderr

        self.kv_storage = dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            self.f_stderr.write(format('\n'.join(format_exception(exc_type, exc_value, exc_traceback))))

        self.f_stdout.close()
        self.f_stderr.close()
        
        copytree(self.tmp_dirname, self.log_dirname)

        shutil.rmtree(self.tmp_dirname)

        with open(os.path.join(self.log_dirname, 'kv_storage.pkl'), 'wb') as f:
            _pickle.dump(self.kv_storage, f)

    def put(self, key, value):
        assert is_picklable(key)
        assert is_picklable(value)

        self.kv_storage[key] = value

