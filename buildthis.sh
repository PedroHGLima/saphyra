#!/bin/bash
#source setup_module.sh
#source setup_module.sh --head
export Athena_SETUP=off
mkdir build
cd build
cmake3 ..
make -j4
cd ..


