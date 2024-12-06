#!/bin/bash

for file in ../build/bin/*; do
    # extract filename from path
    filename=$(basename $file)

    echo -e "\tRunning $filename"
    $file "results/$filename"
done