@rem /** ---- Docs ----
@rem  * model=CONTINUOUS_SPIN_MODEL|UP_DOWN_MODEL
@rem  * randomize=0|1
@rem  * startAtMaxB=0|1
@rem  * mol_type=LINEAR|CIRCULAR|__PATH__.mmb
@rem  */

@rem // ---- Edit Here ----
@set model=CONTINUOUS_SPIN_MODEL
@set randomize=0
@set startAtMaxB=0
@set mol_type=LINEAR

@set out_head=magnetization2




@rem ------ Don't Edit Below This ------
@rem -- Try to change to the project folder on the L: drive, stored in %projDir%
@rem -- %projDir% holds the full path (including the drive letter) to the desired project folder
@rem -- %trgDir% holds the name (path suffix) of the working directory, in case we are in a local copy of the folder
@set projDir=L:\MSD Research Project (v2.1.1)
@if "%cd%" == "%projDir%" goto STAY
@set trgDir=MSD Research Project (v2.1.1)
@set dirTail=%cd%
:REMOVE_CHAR
@if "%dirTail%" == "%trgDir%" goto FIN_TAIL
@set dirTail=%dirTail:~1%
@if not defined dirTail goto FIN_TAIL
@goto REMOVE_CHAR
:FIN_TAIL
@if defined dirTail goto STAY
@if not exist "%projDir%" goto STAY
@%projDir:~0,2%
@cd "%projDir%"
:STAY

@rem -- Find the next open file name
@set prgm=magnetize2
@set id=0
:INC_ID
@set /a id=%id%+1
@if %id% LEQ 0 goto STOP
@set out_file="out\%out_head%, %date:~4,2%-%date:~7,2%-%date:~10,4%, %id%.csv"
@if exist %out_file% goto INC_ID

@rem -- Run the program
@date /t
@time /t
@echo ----------------------------------------
bin\%prgm% %out_file% %model% %randomize% %startAtMaxB% %mol_type%
@echo ----------------------------------------
@date /t
@time /t
@goto DONE

:STOP
@echo Error (%out_file%): No more possible file names exist!

:DONE
@pause
