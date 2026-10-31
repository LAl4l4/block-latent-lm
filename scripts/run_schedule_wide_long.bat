@echo off
rem Scaling check with enough data: wide model (d=512, 8 layers) for 20k steps, causal vs order-only reverse16g8.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-mask-wide-long
set LOG=runs\schedule-mask\overnight-status.txt
if not exist %OUT% mkdir %OUT%
call :run causal-s42 --schedule causal
call :run reverse16g8-s42 --schedule reverse --stagger 16 --segment 8
%PY% -m seam.summarize %OUT% > %OUT%\summary.md 2>&1
echo [%date% %time%] wide long all done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin wide long %1 >> %LOG%
%PY% -m seam.train_mask --out %OUT% --dim 512 --layers 8 --heads 8 --ff-dim 1536 --steps 20000 --eval-every 5000 %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED wide long %1 >> %LOG%) else (echo [%date% %time%] finished wide long %1 >> %LOG%)
goto :eof
