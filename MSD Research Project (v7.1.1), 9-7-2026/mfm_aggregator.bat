@rem -- Find the next open file name
@set id=0
:INC_ID
@set /a id=%id%+1
@if %id% LEQ 0 goto STOP
@set out_file="out\mfm_aggregator, %date:~4,2%-%date:~7,2%-%date:~10,4%, %id%.csv"
@if exist %out_file% goto INC_ID

@date /t
@time /t
@echo ----------------------------------------
bin\mfm_aggregator %out_file%
@echo ----------------------------------------
@date /t
@time /t
@goto DONE

:STOP
@echo Error (%out_file%): No more possible file names exist!

:DONE
@pause
