@echo off
rem Sample eval for long (20k) and wide runs, cross-judged by the other set's causal model. Runs after the capacity check.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set LOG=runs\schedule-mask\overnight-status.txt
:wait
findstr /c:"wide all done" %LOG% >nul 2>&1
if errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto wait
)
echo [%date% %time%] begin long sample eval >> %LOG%
%PY% -m seam.sample_eval --root runs\schedule-mask-long --judge ..\schedule-mask-wide\causal-s42 --runs causal-s42 reverse16-s42 reverse16g8-s42 reverse8g8-s42 forward8-s42 --out runs\schedule-mask-long\sample-eval-judge-wide.json > runs\schedule-mask-long\sample-eval.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED long sample eval >> %LOG%) else (echo [%date% %time%] finished long sample eval >> %LOG%)
echo [%date% %time%] begin wide sample eval >> %LOG%
%PY% -m seam.sample_eval --root runs\schedule-mask-wide --judge ..\schedule-mask-long\causal-s42 --runs causal-s42 reverse16g8-s42 reverse8g8-s42 forward8-s42 --out runs\schedule-mask-wide\sample-eval-judge-long.json > runs\schedule-mask-wide\sample-eval.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED wide sample eval >> %LOG%) else (echo [%date% %time%] finished wide sample eval >> %LOG%)
echo [%date% %time%] samples all done >> %LOG%
