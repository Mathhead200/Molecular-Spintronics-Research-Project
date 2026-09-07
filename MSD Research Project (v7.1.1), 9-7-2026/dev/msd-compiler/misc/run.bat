@if "%1"=="" (
	echo "Run which build?"
	exit /b
)

@set name=%1
"bin/%name%.exe" || exit /b
