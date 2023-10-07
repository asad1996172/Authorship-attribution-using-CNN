#!/bin/bash

# Get all python files not in .gitignore
files=$(git ls-files '*.py' ':!:*.gitignore')

for file in $files; do
    echo "Formatting $file"
    isort "$file"
    black --line-length 88 "$file"
    flake8 --max-line-length=88 "$file"
done