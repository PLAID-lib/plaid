#!/bin/bash
for file in *.py utils/*.py containers/*.py
    do echo "--------------------------------------------------------------------------------------"
        echo "#---# run python $file"
        python $file
    done
