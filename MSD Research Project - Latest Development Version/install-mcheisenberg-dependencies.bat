@echo off
rem Must install Visual Studio (not VS Code) with x64 build tools
rem Must install python (add to PATH)

rem used in runtime_executors.py, and some apps
pip install psutil

rem Used extensively in the simulation sub-package. Also a dependancy for any apps using pandas.
pip install numpy
pip install tqdm

rem TODO: used in plot sub-package
rem TODO: matplotlib

rem For build_ln which uses remez, also some apps
pip install scipy

rem Used in some apps.
pip install pandas
pip install xlsxwriter
pip install openpyxl

pause
