@echo off
rem Speed vs jumps trade-off: stagger s = 4, 2, 1 (parallelism 16/s in steady state), small model, 5k steps.
rem reverse s with g=s (segment reversal), reverse s with g=1 (token reversal) and forward s at the same speed.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-mask-tradeoff
set LOG=runs\schedule-mask-tradeoff\status.txt
if not exist %OUT% mkdir %OUT%
echo [%date% %time%] start >> %LOG%
call :run causal-s42 --schedule causal
call :run reverse4g4-s42 --schedule reverse --stagger 4 --segment 4
call :run reverse2g2-s42 --schedule reverse --stagger 2 --segment 2
call :run reverse2-s42 --schedule reverse --stagger 2
call :run reverse1-s42 --schedule reverse --stagger 1
call :run forward2-s42 --schedule forward --stagger 2
call :run forward1-s42 --schedule forward --stagger 1
%PY% -m stage2.mask.summarize %OUT% > %OUT%\summary.md 2>&1
echo [%date% %time%] tradeoff all done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin %1 >> %LOG%
%PY% -m stage2.mask.train --out %OUT% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
