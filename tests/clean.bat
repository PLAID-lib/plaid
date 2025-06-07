@echo off
setlocal

call :erase "tests\post\converge_bisect_plot.png"
call :erase "tests\post\differ_bisect_plot.png"
call :erase "tests\post\equal_bisect_plot.png"
call :erase "tests\post\first_metrics.yaml"
call :erase "tests\post\second_metrics.yaml"
call :erase "tests\post\third_metrics.yaml"

goto :eof

:erase
if exist %1 (
    del %1
) else (
    echo Unknown file %1
)
goto :eof