#!/bin/bash

erase () {
    if [ -f "$1" ]; then
        rm "$1"
    else
        echo "Unknown file $1"
    fi
}

erase "tests/post/converge_bisect_plot.png"
erase "tests/post/differ_bisect_plot.png"
erase "tests/post/equal_bisect_plot.png"
erase "tests/post/first_metrics.yaml"
erase "tests/post/second_metrics.yaml"
erase "tests/post/third_metrics.yaml"