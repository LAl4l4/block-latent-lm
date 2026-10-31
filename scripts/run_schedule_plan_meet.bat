@echo off
rem Meet-in-the-middle with a block plan. Waits for run_schedule_meet.bat.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-plan
set LOG=runs\schedule-plan\status.txt
:wait
findstr /c:"meet all done" runs\schedule-mask\overnight-status.txt >nul 2>&1
if errorlevel 1 (
  ping -n 16 127.0.0.1 >nul
  goto wait
)
call :run meet1-plan4x16-s42 --schedule meet --stagger 1
call :run meet4-plan4x16-s42 --schedule meet --stagger 4
echo [%date% %time%] plan meet done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin %1 >> %LOG%
%PY% -m seam.train_plan --out %OUT% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
