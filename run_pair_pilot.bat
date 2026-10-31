@echo off
rem Pair-lookahead pilot: run tests, then four 3000-step arms sequentially.
rem Logs and results: runs\pair-pilot\ ; progress: runs\pair-pilot\status.txt
rem Order follows docs\paper-plan.md step 1: sync vs delayed (const) first, then joint Q.
rem All arms start from the frozen, hashed teacher.pt.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\pair-pilot
set INIT=--init runs\feedback-distill-greedy\data\teacher.pt
if not exist %OUT% mkdir %OUT%
echo [%date% %time%] start >> %OUT%\status.txt

%PY% -m pytest -q tests\test_pair_training.py tests\test_pair_feedback.py tests\test_calm_token.py > %OUT%\pytest.log 2>&1
if errorlevel 1 goto testfail
echo [%date% %time%] pytest passed >> %OUT%\status.txt

call :run sync-s42 --mode sync %INIT%
call :run const-all-s42 --mode const --train all %INIT%
call :run state-all-s42 --mode state --train all %INIT%
call :run state-student-s42 --mode state --train student %INIT%

%PY% -m experiments.summarize_pair_pilot %OUT% > %OUT%\summary.md 2>&1
echo [%date% %time%] all done >> %OUT%\status.txt
goto :eof

:testfail
echo [%date% %time%] pytest FAILED, see pytest.log >> %OUT%\status.txt
goto :eof

:run
set NAME=%1
echo [%date% %time%] begin %NAME% >> %OUT%\status.txt
%PY% -m experiments.train_pair_lookahead --run-dir %OUT%\%NAME% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%NAME%.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] FAILED %NAME% >> %OUT%\status.txt
) else (
  echo [%date% %time%] finished %NAME% >> %OUT%\status.txt
)
goto :eof
