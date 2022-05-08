import os
from os import remove, listdir
from os.path import join
from shutil import rmtree

from gdeep.utility.constants import ROOT_DIR

def clean_up_files(fun):
    """Decorator to remove all the files created by the method it decorates"""
    def wrapper():
        # Get all subdirectories of the root directory
        directories = [d for d in listdir(ROOT_DIR) if os.path.isdir(join(ROOT_DIR, d))]

        # Get all files in the root directory
        files = [f for f in listdir(ROOT_DIR) if os.path.isfile(join(ROOT_DIR, f))]


        fun()
        # Get all subdirectories of the root directory
        new_directories = [d for d in listdir(ROOT_DIR) if os.path.isdir(join(ROOT_DIR, d))]

        # Get all files in the root directory
        new_files = [f for f in listdir(ROOT_DIR) if os.path.isfile(join(ROOT_DIR, f))]

        # Delete all newly created directories
        for directory in new_directories:
            if directory not in directories:
                rmtree(join(ROOT_DIR, directory))

        # Delete all newly created files
        for file in new_files:
            if file not in files:
                remove(join(ROOT_DIR, file))
    return wrapper