@echo off
rem Rebuild docs\technical_report.pdf after editing docs\technical_report.md (Windows version of build_pdf.sh).
rem Needs pandoc (https://pandoc.org/installing.html) and xelatex (MiKTeX, https://miktex.org/download).
rem Run it from anywhere: double-click it, or type  docs\build_pdf.bat
setlocal

rem pushd (not cd) so it also works on a network drive (\\server\share\...)
pushd "%~dp0"
if not exist technical_report.md (echo Cannot find technical_report.md next to this script & goto :fail)

pandoc --version >nul 2>nul
if errorlevel 1 (echo pandoc not found: install it from https://pandoc.org/installing.html & goto :fail)
xelatex --version >nul 2>nul
if errorlevel 1 (echo xelatex not found: install MiKTeX from https://miktex.org/download & goto :fail)

rem Same fonts as build_pdf.sh when installed (https://dejavu-fonts.github.io); otherwise Windows fonts
set "MAIN=DejaVu Serif"
set "MONO=DejaVu Sans Mono"
if not exist "%WINDIR%\Fonts\DejaVuSerif.ttf" if not exist "%LOCALAPPDATA%\Microsoft\Windows\Fonts\DejaVuSerif.ttf" (
  echo DejaVu fonts not installed: using Cambria and Consolas, so some symbols or line breaks may differ.
  set "MAIN=Cambria"
  set "MONO=Consolas"
)

pandoc technical_report.md -o technical_report.pdf --pdf-engine=xelatex --lua-filter=apa_figures.lua -V mainfont="%MAIN%" -V monofont="%MONO%"
if errorlevel 1 goto :fail
echo written docs\technical_report.pdf
popd
exit /b 0

:fail
echo PDF not built.
popd 2>nul
exit /b 1
