@echo off
rem Capacity check: does the ordering cost shrink with a wider model? Runs after the overnight batch.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set WIDE=runs\schedule-mask-wide
set LOG=runs\schedule-mask\overnight-status.txt
if not exist %WIDE% mkdir %WIDE%
:wait
findstr /c:"all done" %LOG% >nul 2>&1
if errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto wait
)
call :run causal-s42 --schedule causal
call :run reverse16g8-s42 --schedule reverse --stagger 16 --segment 8
call :run reverse8g8-s42 --schedule reverse --stagger 8 --segment 8
call :run forward8-s42 --schedule forward --stagger 8
%PY% -m stage2.mask.summarize %WIDE% > %WIDE%\summary.md 2>&1
echo [%date% %time%] wide all done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin wide %1 >> %LOG%
%PY% -m stage2.mask.train --out %WIDE% --dim 512 --layers 8 --heads 8 --ff-dim 1536 %2 %3 %4 %5 %6 %7 %8 %9 > %WIDE%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED wide %1 >> %LOG%) else (echo [%date% %time%] finished wide %1 >> %LOG%)
goto :eof
