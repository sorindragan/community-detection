#!/bin/bash
args=("$@")
python setup.py build_ext --inplace
python run_compiled.py ${args[0]}
