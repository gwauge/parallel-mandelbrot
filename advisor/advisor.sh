#!/bin/bash

for file in ../build/bin/*; do
    # extract filename from path
    filename=$(basename $file)

    echo -e "\tRunning $filename"
    /opt/intel/oneapi/advisor/2024.1/bin64/advisor -collect roofline -trip-counts -flop -module-filter-mode=exclude -mrte-mode=auto -interval=10 -data-limit=500 -stackwalk-mode=offline -stack-unwind-limit=8388608 -stack-stitching -mkl-user-mode -no-profile-python -no-support-multi-isa-binaries -no-spill-analysis -no-static-instruction-mix -auto-finalize -show-report -no-profile-gpu -gpu-sampling-interval=1 -profile-intel-perf-libs --app-working-dir="$(pwd)/working_dir" --project-dir="./${filename}" -- "$(pwd)/../build/bin/${filename}" "${filename}"
done
