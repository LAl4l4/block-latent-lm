@echo off
rem Window decoder with gated left-attention and robustness training (z noise, word corruption).
rem 5 arms x capacity 64/128. Finished runs are skipped, unfinished ones retrained.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-window
set LOG=runs\schedule-window\status.txt
set GR=--gate --robust
if not exist %OUT% mkdir %OUT%
echo [%date% %time%] start gate+robust sweep >> %LOG%
for %%c in (64 128) do (
  call :run causal-cap%%c-w0-gate-robust-s42 --capacity %%c --schedule causal --window 0
  call :run causal-cap%%c-w2-gate-robust-s42 --capacity %%c --schedule causal
  call :run forward8-cap%%c-w2-gate-robust-s42 --capacity %%c --schedule forward --stagger 8
  call :run reverse8-cap%%c-w2-gate-robust-s42 --capacity %%c --schedule reverse --stagger 8
  call :run reverse8g8-cap%%c-w2-gate-robust-s42 --capacity %%c --schedule reverse --stagger 8 --segment 8
)
echo [%date% %time%] gate+robust sweep done >> %LOG%
goto :eof
:run
if exist %OUT%\%1\final.json (echo [%date% %time%] skip finished %1 >> %LOG% & goto :eof)
if exist %OUT%\%1 rmdir /s /q %OUT%\%1
echo [%date% %time%] begin %1 >> %LOG%
%PY% -m stage3.window.train --out %OUT% %GR% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
