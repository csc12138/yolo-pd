@echo off
setlocal EnableDelayedExpansion
for %%n in (*.jpg) do (
set "name=%%n"
set "name=!name:a (=!"
set "name=!name:)=!"
ren "%%n" "!name!"
)