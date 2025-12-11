@rem /** ---- Docs ----
@rem  * model=CONTINUOUS_SPIN_MODEL|UP_DOWN_MODEL
@rem  * mode=RANDOMIZE|REINITIALIZE
@rem  * mol_type=LINEAR|CIRCULAR|__PATH__.mmb
@rem  * threadCount=<uint32 >= 1>
@rem  */


@rem // ---- Edit Here ----
@set model=CONTINUOUS_SPIN_MODEL
@set mode=RANDOMIZE
@set mol_type=LINEAR
@set threadCount=3

@set paramFile=parameters-metropolis.txt
@set out_head=metropolis



@rem ------ Don't Edit Below This ------
@rem -- Find the next open file name
@set prgm=metropolis
@set id=0
:INC_ID
@set /a id=%id%+1
@if %id% LEQ 0 goto STOP
@set out_file="out\%out_head%, %date:~4,2%-%date:~7,2%-%date:~10,4%, %id%.xml"
@if exist %out_file% goto INC_ID

@date /t
@time /t
@echo ----------------------------------------
bin\%prgm% %paramFile% %out_file% %model% %mode% %mol_type% %threadCount%
@echo ----------------------------------------
@date /t
@time /t
@goto DONE

:STOP
@echo Error (%out_file%): No more possible file names exist!

:DONE
@pause
