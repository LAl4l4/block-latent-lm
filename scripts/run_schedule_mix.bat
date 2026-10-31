@echo off
rem Does sharing the model with causal training (mixed masks) reduce the ordering cost? Runs after the queued sample evals.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-mask
set LOG=runs\schedule-mask\overnight-status.txt
:wait
findstr /c:"samples all done" %LOG% >nul 2>&1
if errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto wait
)
call :run reverse16g8mix50-s42 --schedule reverse --stagger 16 --segment 8 --mix-causal 0.5
call :run reverse8g8mix50-s42 --schedule reverse --stagger 8 --segment 8 --mix-causal 0.5
call :run reverse8g8mix25-s42 --schedule reverse --stagger 8 --segment 8 --mix-causal 0.25
call :run reverse0mix50-s42 --schedule reverse --stagger 0 --mix-causal 0.5
%PY% -m seam.summarize %OUT% > %OUT%\summary.md 2>&1
echo [%date% %time%] mix all done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin %1 >> %LOG%
%PY% -m seam.train_mask --out %OUT% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
