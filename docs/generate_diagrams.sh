#!/bin/bash

#---# Clean
rm -rf ../public/diagrams
mkdir -p ../public/diagrams
cd ../public/diagrams

#---# Make class and package diagrams for plaid
mkdir plaid
cd plaid
pyreverse --colorize -o html ../../../src/plaid
pyreverse --colorize -o dot ../../../src/plaid
cd ..

# #---# Make class and package diagrams for tests
# mkdir tests
# cd tests
# pyreverse --colorize -o html ../../../tests
# pyreverse --colorize -o dot ../../../tests
# cd ..
# #---# Make class and package diagrams for examples
# mkdir examples
# cd examples
# pyreverse --colorize -o html ../../../examples
# pyreverse --colorize -o dot ../../../examples
# cd ..
