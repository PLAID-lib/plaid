#!/bin/bash

if [[ "$(uname)" == "Linux" ]]; then
    FILES="*.py examples/*.py bridges/*.py utils/*.py containers/*.py post/*.py pipelines/*.py"
else
    FILES="*.py examples/*.py utils/*.py containers/*.py post/*.py"
fi

for file in $FILES
do
    echo "--------------------------------------------------------------------------------------"
    echo "#---# run python $file"
    python "$file"
    if [ $? -ne 0 ]; then
        exit 1
    fi
done