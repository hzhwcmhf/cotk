@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build
set SPHINXPROJ=cotk
set SPHINXOPTS=-W

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

if "%1" == "public" goto public

if "%1" == "internal" goto internal

if "%1" == "checkpublic" goto checkpublic

if "%1" == "public_only_source" goto public_only_source

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:public

cd meta && python update_doc.py -D public && cd ..
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:internal

cd meta && python update_doc.py -D internal && cd ..
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:checkpublic

cd meta && python update_doc.py --check -D public && cd ..
goto end

:public_only_source

cd meta && python update_doc.py -D public --only_source && cd ..
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
