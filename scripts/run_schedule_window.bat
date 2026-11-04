@echo off
rem CALM-style strong z + small window decoder (own z, written words of 2 left blocks).
rem 6 arms x capacity 64/128/32 nats per block. Finished runs (final.json) are skipped;
rem an unfinished run directory is deleted and retrained, so the sweep can be resumed.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-window
set LOG=runs\schedule-window\status.txt
if not exist %OUT% mkdir %OUT%
echo [%date% %time%] start window sweep >> %LOG%
for %%c in (64 128 32) do (
  call :run zonly-cap%%c-w2-s42 --capacity %%c --schedule zonly
  call :run causal-cap%%c-w2-s42 --capacity %%c --schedule causal
  call :run forward8-cap%%c-w2-s42 --capacity %%c --schedule forward --stagger 8
  call :run reverse8-cap%%c-w2-s42 --capacity %%c --schedule reverse --stagger 8
  call :run reverse8g8-cap%%c-w2-s42 --capacity %%c --schedule reverse --stagger 8 --segment 8
  call :run causal-cap%%c-w0-s42 --capacity %%c --schedule causal --window 0
)
echo [%date% %time%] window sweep done >> %LOG%
goto :eof
:run
if exist %OUT%\%1\final.json (echo [%date% %time%] skip finished %1 >> %LOG% & goto :eof)
if exist %OUT%\%1 rmdir /s /q %OUT%\%1
echo [%date% %time%] begin %1 >> %LOG%
%PY% -m seam.train_window --out %OUT% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
