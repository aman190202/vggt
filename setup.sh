#!/bin/bash
set -e

pip install -r requirements.txt
pip install -r requirements_demo.txt
pip install -e .
pip install tensorboard wcmatch fvcore iopath nvitop