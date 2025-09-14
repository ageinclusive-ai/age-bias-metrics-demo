@echo off
setlocal enabledelayedexpansion
REM ==========================================================
REM One-click batch run (CSV mode) for Bias Checker
REM Edit the CONFIG section below, then double-click this file.
REM Requires: Python, pandas, requests. LM Studio server running.
REM ==========================================================

REM -------- CONFIG --------
set "ADS_CSV=adverts.csv"
set "AD_COLUMN=ad_text"
set "ID_COLUMN=ad_id"
set "SYSTEM_PROMPT=system_prompt.txt"
set "LEXICON=lexicon.csv"
set "ENDPOINT=http://localhost:1234/v1"
set "MODEL=meta-llama-3.1-8b-instruct"
set "OUTDIR=results"
REM Uncomment to use a virtual environment:
REM call "venv\Scripts\activate"

echo.
echo [Bias Checker] Running batch (CSV mode) ...
echo   ADS_CSV      = "%ADS_CSV%"
echo   AD_COLUMN    = "%AD_COLUMN%"
echo   ID_COLUMN    = "%ID_COLUMN%"
echo   SYSTEM_PROMPT= "%SYSTEM_PROMPT%"
echo   LEXICON      = "%LEXICON%"
echo   ENDPOINT     = "%ENDPOINT%"
echo   MODEL        = "%MODEL%"
echo   OUTDIR       = "%OUTDIR%"
echo.

python "%~dp0batch_runner.py" ^
  --ads-csv "%ADS_CSV%" ^
  --ad-column "%AD_COLUMN%" ^
  --id-column "%ID_COLUMN%" ^
  --system-prompt "%SYSTEM_PROMPT%" ^
  --lexicon "%LEXICON%" ^
  --endpoint "%ENDPOINT%" ^
  --model "%MODEL%" ^
  --outdir "%OUTDIR%"

if errorlevel 1 (
  echo.
  echo [ERROR] Batch run failed. See messages above.
  pause
  exit /b 1
)

echo.
echo [OK] Done. Opening results folder...
if exist "%OUTDIR%" (
  start "" explorer "%OUTDIR%"
) else (
  echo Results folder not found: "%OUTDIR%"
)
echo.
pause
