@echo off
rem Overnight schedule study (docs\idea-reverse-staggered-decoding.md).
rem Waits for run_schedule_mask.bat, then: B new schedules (seed 42), E sample eval,
rem C seeds 43/44 for core arms, D 20000-step training to test learnability.
rem Progress: runs\schedule-mask\overnight-status.txt
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-mask
set LONG=runs\schedule-mask-long
set LOG=%OUT%\overnight-status.txt
if not exist %LONG% mkdir %LONG%

:wait
findstr /c:"all done" %OUT%\status.txt >nul 2>&1
if errorlevel 1 (
  ping -n 31 127.0.0.1 >nul
  goto wait
)
echo [%date% %time%] start >> %LOG%
%PY% -m pytest -q tests\test_schedule_mask_study.py > %OUT%\overnight-pytest.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] pytest FAILED >> %LOG%
  goto :eof
)

rem B: segment-reversed and fully parallel schedules, seed 42
call :run reverse8g8-s42 --schedule reverse --stagger 8 --segment 8
call :run reverse8g4-s42 --schedule reverse --stagger 8 --segment 4
call :run reverse16g8-s42 --schedule reverse --stagger 16 --segment 8
call :run reverse16g4-s42 --schedule reverse --stagger 16 --segment 4
call :run reverse0-s42 --schedule reverse --stagger 0
call :run reverse0g4-s42 --schedule reverse --stagger 0 --segment 4
call :run forward0-s42 --schedule forward --stagger 0
call :run landing-s42 --schedule landing

rem E: sample quality, judged by causal-s42
echo [%date% %time%] begin sample eval >> %LOG%
%PY% -m seam.mask.sample_eval --runs causal-s42 delayed-s42 forward8-s42 reverse8-s42 reverse8g8-s42 reverse8g4-s42 reverse16-s42 reverse16g8-s42 reverse4-s42 forward4-s42 reverse0-s42 reverse0g4-s42 forward0-s42 landing-s42 --out %OUT%\sample-eval-s42.json > %OUT%\sample-eval.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED sample eval >> %LOG%) else (echo [%date% %time%] finished sample eval >> %LOG%)

rem C: seeds 43 and 44 for core arms
for %%S in (43 44) do (
  call :run causal-s%%S --schedule causal --seed %%S
  call :run reverse8-s%%S --schedule reverse --stagger 8 --seed %%S
  call :run forward8-s%%S --schedule forward --stagger 8 --seed %%S
  call :run reverse8g8-s%%S --schedule reverse --stagger 8 --segment 8 --seed %%S
  call :run reverse16-s%%S --schedule reverse --stagger 16 --seed %%S
  call :run reverse16g8-s%%S --schedule reverse --stagger 16 --segment 8 --seed %%S
  call :run delayed-s%%S --schedule delayed --seed %%S
  call :run reverse0-s%%S --schedule reverse --stagger 0 --seed %%S
  call :run forward0-s%%S --schedule forward --stagger 0 --seed %%S
  call :run landing-s%%S --schedule landing --seed %%S
)
%PY% -m seam.mask.summarize %OUT% > %OUT%\summary.md 2>&1

rem D: 20000 steps, does the order (learnability) cost shrink with training?
call :runlong causal-s42 --schedule causal
call :runlong reverse16-s42 --schedule reverse --stagger 16
call :runlong reverse16g8-s42 --schedule reverse --stagger 16 --segment 8
call :runlong reverse8g8-s42 --schedule reverse --stagger 8 --segment 8
call :runlong forward8-s42 --schedule forward --stagger 8
%PY% -m seam.mask.summarize %LONG% > %LONG%\summary.md 2>&1
echo [%date% %time%] all done >> %LOG%
goto :eof

:run
echo [%date% %time%] begin %1 >> %LOG%
%PY% -m seam.mask.train --out %OUT% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof

:runlong
echo [%date% %time%] begin long %1 >> %LOG%
%PY% -m seam.mask.train --out %LONG% --steps 20000 --eval-every 5000 %2 %3 %4 %5 %6 %7 %8 %9 > %LONG%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED long %1 >> %LOG%) else (echo [%date% %time%] finished long %1 >> %LOG%)
goto :eof
