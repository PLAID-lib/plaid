#!/bin/bash

# Too much arguments provided
if [ $# -gt 1 ]; then
    echo "Usage: <path>"
    exit 1
fi

path="$1"
# Check if the path exists
if [ $# -eq 1 ] && [ ! -e "$path" ]; then
    echo "Path '$path' does not exist. Abort."
    exit 2
fi

# Search in the entire project from 'path'
python_files=($(find ./$path -type f -name "*.py"))
length="${#python_files[*]}"

VAR=0
echo "Start formatting $length file(s)."

for file in "${python_files[@]}"; do
    VAR=$(($VAR + 1))
    echo -n "$VAR/$length formatting $file."

    # Autopep8 format convention
    autopep8 --in-place --aggressive "$file"; echo -n '.'
    isort -q "$file"; echo ''
    #black --quiet --line-length 80 "$file"; echo -n '.'
done

echo "Formatting complete. $length file(s) checked"