#!/bin/bash

if [[ "$(uname)" == "Linux" ]]; then
    FILES="*.py utils/*.py containers/*.py bridges/*.py"
else
    FILES="*.py utils/*.py containers/*.py"
fi

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

for file in $FILES
do
    echo "--------------------------------------------------------------------------------------"
    echo "#---# run python $file"
    python "$file"
    if [ $? -ne 0 ]; then
        exit 1
    fi
done