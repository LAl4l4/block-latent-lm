@echo off
rem 1) Refine study (small model, 5k steps): re-decode the jump token (reverse8g8) or the block head (forward8)
rem    once the block is written; samples judged by an independently trained causal model (seed 43).
rem 2) Scaling check without overfitting: wide model 20k steps on 1M stories (219M tokens, same tokenizer and dev set).
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-mask-refine
set LOG=runs\schedule-mask-refine\status.txt
set EXTRA=
if not exist %OUT% mkdir %OUT%
echo [%date% %time%] start >> %LOG%
call :run causal-s43 --schedule causal --seed 43
call :run reverse8g8refine-s42 --schedule reverse --stagger 8 --segment 8 --refine
call :run forward8refine-s42 --schedule forward --stagger 8 --refine
echo [%date% %time%] begin samples >> %LOG%
%PY% -m seam.sample_eval --root %OUT% --runs reverse8g8refine-s42 forward8refine-s42 --judge causal-s43 --out %OUT%\sample-eval.json > %OUT%\sample-eval.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED samples >> %LOG%) else (echo [%date% %time%] finished samples >> %LOG%)
echo [%date% %time%] refine all done >> %LOG%
set OUT=runs\schedule-mask-wide-1m
if not exist %OUT% mkdir %OUT%
set EXTRA=--data data\prepared-1m --dim 512 --layers 8 --heads 8 --ff-dim 1536 --steps 20000 --eval-every 5000
call :run causal-s42 --schedule causal
call :run reverse16g8-s42 --schedule reverse --stagger 16 --segment 8
%PY% -m seam.summarize %OUT% > %OUT%\summary.md 2>&1
echo [%date% %time%] wide 1m all done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin %OUT% %1 >> %LOG%
%PY% -m seam.train_mask --out %OUT% %EXTRA% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
