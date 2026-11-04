@echo off
rem User's original scheme: pairs of blocks decoded together, forward + reverse, meeting at the seam.
cd /d D:\Code\latent-ar-async-decoder
set PY=D:\Code\venv313\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OUT=runs\schedule-mask
set LOG=runs\schedule-mask\overnight-status.txt
call :run meet1-s42 --schedule meet --stagger 1
call :run meet4-s42 --schedule meet --stagger 4
call :run alternate1-s42 --schedule alternate --stagger 1
%PY% -m stage2.mask.summarize %OUT% > %OUT%\summary.md 2>&1
echo [%date% %time%] meet all done >> %LOG%
goto :eof
:run
echo [%date% %time%] begin %1 >> %LOG%
%PY% -m stage2.mask.train --out %OUT% %2 %3 %4 %5 %6 %7 %8 %9 > %OUT%\%1.log 2>&1
if errorlevel 1 (echo [%date% %time%] FAILED %1 >> %LOG%) else (echo [%date% %time%] finished %1 >> %LOG%)
goto :eof
