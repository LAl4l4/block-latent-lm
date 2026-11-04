@echo off
rem Schedule-cost study (docs\idea-reverse-staggered-decoding.md, validation step 1).
rem Token-level two-stream LM; only the attention mask differs between arms.
rem Logs and results: runs\schedule-mask\ ; progress: runs\schedule-mask\status.txt
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-mask
if not exist %OUT% mkdir %OUT%
echo [%date% %time%] start >> %OUT%\status.txt

%PY% -m pytest -q tests\test_schedule_mask_study.py > %OUT%\pytest.log 2>&1
if errorlevel 1 goto testfail
echo [%date% %time%] pytest passed >> %OUT%\status.txt

call :run causal-s42 --schedule causal
call :run reverse8-s42 --schedule reverse --stagger 8
call :run forward8-s42 --schedule forward --stagger 8
call :run delayed-s42 --schedule delayed
call :run reverse16-s42 --schedule reverse --stagger 16
call :run reverse4-s42 --schedule reverse --stagger 4
call :run forward4-s42 --schedule forward --stagger 4

%PY% -m seam.mask.summarize %OUT% > %OUT%\summary.md 2>&1
echo [%date% %time%] all done >> %OUT%\status.txt
goto :eof

:testfail
echo [%date% %time%] pytest FAILED, see pytest.log >> %OUT%\status.txt
goto :eof

:run
set NAME=%1
echo [%date% %time%] begin %NAME% >> %OUT%\status.txt
%PY% -m seam.mask.train --out %OUT% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%NAME%.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] FAILED %NAME% >> %OUT%\status.txt
) else (
  echo [%date% %time%] finished %NAME% >> %OUT%\status.txt
)
goto :eof
