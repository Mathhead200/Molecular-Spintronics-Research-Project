@echo off
rem Must install Visual Studio (not VS Code) with x64 build tools
rem Must install python (add to PATH)

pip install numpy
pip install pandas
pip install xlsxwriter
rem TODO: matplotlib
pip install tqdm
pip install psutil

rem For build_ln which uses remez
rem @pip install scipy

pause
