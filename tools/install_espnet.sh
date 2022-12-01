#!/bin/bash
rm -rf espnet
git clone https://github.com/espnet/espnet

# The espent editable import to HYnet
. ./activate_python.sh && cd espnet/tools && python3 -m pip install -e "..[recipe]"
