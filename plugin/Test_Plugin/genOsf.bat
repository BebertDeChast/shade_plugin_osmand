@echo off
setlocal enabledelayedexpansion

cd "Test_Plugin"

REM Nom du dossier courant
for %%A in ("%cd%") do set folderName=%%~nA

del "..\%folderName%.osf"

REM Chemin de sortie dans le dossier parent
set outputZip="..\\%folderName%.zip"

REM Création de l'archive via PowerShell
powershell -command "Compress-Archive -Path * -DestinationPath "..\%folderName%.zip" -Force"

ren "..\%folderName%.zip" "%folderName%.osf"

pause