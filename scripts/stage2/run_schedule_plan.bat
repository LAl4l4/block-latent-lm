@echo off
rem Plan-conditioned schedules: does a per-block discrete plan (4 codes x 16) absorb the ordering cost?
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-plan
set LOG=runs\schedule-plan\status.txt
if not exist %OUT% mkdir %OUT%
echo [%date% %time%] start >> %LOG%
call :run causal-plan4x16-s42 --schedule causal
call :run reverse16g8-plan4x16-s42 --schedule reverse --stagger 16 --segment 8
call :run reverse8g8-plan4x16-s42 --schedule reverse --stagger 8 --segment 8
call :run forward8-plan4x16-s42 --schedule forward --stagger 8
echo [%date% %time%] plan all done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin %1 >> %LOG%
%PY% -m seam.plan.train --out %OUT% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
