@echo off
rem Continuous block latents (VAE, 4 slots x 16 dims): small model 5k steps, then wide model 20k steps.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-latent
set LOG=runs\schedule-latent\status.txt
set EXTRA=
if not exist %OUT% mkdir %OUT%
echo [%date% %time%] start >> %LOG%
call :run causal-latent4x16-s42 --schedule causal
call :run reverse8g8-latent4x16-s42 --schedule reverse --stagger 8 --segment 8
call :run reverse16-latent4x16-s42 --schedule reverse --stagger 16
call :run forward8-latent4x16-s42 --schedule forward --stagger 8
echo [%date% %time%] latent small done >> %LOG%
set OUT=runs\schedule-latent-wide-long
if not exist %OUT% mkdir %OUT%
set WIDE=--dim 512 --layers 8 --heads 8 --ff-dim 1536 --steps 20000 --eval-every 5000
set EXTRA=%WIDE%
call :run causal-latent4x16-s42 --schedule causal
call :run reverse8g8-latent4x16-s42 --schedule reverse --stagger 8 --segment 8
echo [%date% %time%] latent wide done >> %LOG%
rem Order-only wide reverse8g8 (no latent) at 20k, to separate the latent gain from scale.
set OUT=runs\schedule-mask-wide-long-r8
if not exist %OUT% mkdir %OUT%
echo [%date% %time%] begin wide mask reverse8g8-s42 >> %LOG%
%PY% -m seam.train_mask --out %OUT% %WIDE% --schedule reverse --stagger 8 --segment 8 > %OUT%\reverse8g8-s42.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED wide mask reverse8g8-s42 >> %LOG%) else (echo [%date% %time%] finished wide mask reverse8g8-s42 >> %LOG%)
echo [%date% %time%] latent all done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin %OUT% %1 >> %LOG%
%PY% -m seam.train_latent --out %OUT% %EXTRA% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
