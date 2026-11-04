@echo off
rem Token-level reversal (g=1) with a block plan: does a plan help g=1 more than g=8?
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-plan
set LOG=runs\schedule-plan\status.txt
echo [%date% %time%] start g1 >> %LOG%
call :run reverse16-plan4x16-s42 --schedule reverse --stagger 16
call :run reverse8-plan4x16-s42 --schedule reverse --stagger 8
echo [%date% %time%] plan g1 done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin %1 >> %LOG%
%PY% -m stage2.plan.train --out %OUT% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
