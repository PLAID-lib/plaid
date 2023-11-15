#!/bin/bash
for file in *.py utils/*.py containers/*.py plot/*.py
    do echo "--------------------------------------------------------------------------------------"
        echo "#---# run python $file"
        python $file
        if [ $? -ne 0 ]; then
            exit 1
        fi
    done
