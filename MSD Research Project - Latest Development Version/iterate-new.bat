@setlocal
@set in="%~1"
@if %in%=="" set in="parameters-iterate.txt"
python -m mcheisenberg.apps.iterate --in %in% --out out --year 2022 --edition Community
@pause
