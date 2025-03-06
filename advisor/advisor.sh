#!/bin/bash
for file in ../build/bin/basic/*; do
    # extract filename from path
    filename=$(basename $file)
    echo -e "\tRunning $filename"
    advisor -collect=roofline -flop -stacks -trip-counts -project-dir "./${filename}" --search-dir src:r="$(pwd)/../src" --app-working-dir="$(pwd)/
working_dir" -- "$(pwd)/../build/bin/basic/${filename}"
    advisor --snapshot --project-dir="./${filename}" --pack --cache-sources --cache-binaries --search-dir src:r="$(pwd)/../src" -- "./${filename}"
    advisor --report=survey --with-stack --format=csv --project-dir="./${filename}" --show-all-columns --report-output="./${filename}.csv"
done
zip results.zip *.csv *.advixeexpz
