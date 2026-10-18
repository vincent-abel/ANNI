@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Native Windows CUDA build for AnnI (MSVC + nvcc + cuBLAS).
REM Run from "x64 Native Tools Command Prompt for VS 2019/2022".
REM Linux: make / make cuda unchanged.

cd /d "%~dp0.."
if errorlevel 1 (
  echo build_windows: cannot cd to repo root
  exit /b 1
)

where cl >nul 2>&1
if errorlevel 1 (
  echo build_windows: cl.exe not found. Open an x64 Native Tools VS prompt.
  exit /b 1
)
where nvcc >nul 2>&1
if errorlevel 1 (
  echo build_windows: nvcc not found. Install CUDA Toolkit and reopen the prompt.
  exit /b 1
)

if "%CUDA_PATH%"=="" (
  for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDA_PATH=%%~D"
)
if "%CUDA_PATH%"=="" (
  echo build_windows: CUDA_PATH not set and toolkit not found under Program Files.
  exit /b 1
)

set "OUTDIR=objects_win"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "INC=/I. /Inet /Inum /Ilang /Icli /Icompat /I"%CUDA_PATH%\include""
set "CFLAGS=/nologo /O2 /W3 /std:c11 /D_CRT_SECURE_NO_WARNINGS /openmp %INC%"

call :compile_all
if errorlevel 1 exit /b 1

echo === compiling num\gpu_cuda.cu ===
nvcc -O2 -DANN_CUBLAS -I. -Inum -Ilang -Icompat -I"%CUDA_PATH%\include" -c num\gpu_cuda.cu -o "%OUTDIR%\gpu_cuda.obj"
if errorlevel 1 (
  echo build_windows: nvcc failed
  exit /b 1
)

echo === linking anni.exe ===
link /nologo /OUT:anni.exe ^
  %OUTDIR%\neuron.obj %OUTDIR%\layer.obj %OUTDIR%\dataset.obj %OUTDIR%\optim.obj %OUTDIR%\serialize.obj ^
  %OUTDIR%\vec.obj %OUTDIR%\mat.obj %OUTDIR%\tensor.obj %OUTDIR%\ops.obj %OUTDIR%\arena.obj %OUTDIR%\simd.obj ^
  %OUTDIR%\tokenize.obj %OUTDIR%\vocab.obj %OUTDIR%\bpe.obj %OUTDIR%\embed.obj %OUTDIR%\posenc.obj ^
  %OUTDIR%\attn.obj %OUTDIR%\ffn.obj %OUTDIR%\layernorm.obj %OUTDIR%\resid.obj %OUTDIR%\block.obj ^
  %OUTDIR%\lm.obj %OUTDIR%\generate.obj %OUTDIR%\gradutil.obj %OUTDIR%\lm_adam.obj ^
  %OUTDIR%\dataset_text.obj %OUTDIR%\dataset_stream.obj %OUTDIR%\lm_serialize.obj ^
  %OUTDIR%\lm_train.obj %OUTDIR%\lm_bench.obj %OUTDIR%\lm_info.obj %OUTDIR%\quant.obj %OUTDIR%\kv.obj ^
  %OUTDIR%\tower.obj %OUTDIR%\chat_session.obj %OUTDIR%\chat_ui.obj ^
  %OUTDIR%\main.obj %OUTDIR%\gpu_cuda.obj ^
  /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cublas.lib
if errorlevel 1 (
  echo build_windows: link failed
  exit /b 1
)

echo built .\anni.exe
exit /b 0

:compile_one
echo   %~1
cl %CFLAGS% /c "%~1" /Fo"%OUTDIR%\%~n1.obj"
if errorlevel 1 (
  echo build_windows: failed compiling %~1
  exit /b 1
)
exit /b 0

:compile_all
echo === compiling C sources ===
call :compile_one net\neuron.c || exit /b 1
call :compile_one net\layer.c || exit /b 1
call :compile_one net\dataset.c || exit /b 1
call :compile_one net\optim.c || exit /b 1
call :compile_one net\serialize.c || exit /b 1
call :compile_one num\vec.c || exit /b 1
call :compile_one num\mat.c || exit /b 1
call :compile_one num\tensor.c || exit /b 1
call :compile_one num\ops.c || exit /b 1
call :compile_one num\arena.c || exit /b 1
call :compile_one num\simd.c || exit /b 1
call :compile_one lang\tokenize.c || exit /b 1
call :compile_one lang\vocab.c || exit /b 1
call :compile_one lang\bpe.c || exit /b 1
call :compile_one lang\embed.c || exit /b 1
call :compile_one lang\posenc.c || exit /b 1
call :compile_one lang\attn.c || exit /b 1
call :compile_one lang\ffn.c || exit /b 1
call :compile_one lang\layernorm.c || exit /b 1
call :compile_one lang\resid.c || exit /b 1
call :compile_one lang\block.c || exit /b 1
call :compile_one lang\lm.c || exit /b 1
call :compile_one lang\generate.c || exit /b 1
call :compile_one lang\gradutil.c || exit /b 1
call :compile_one lang\lm_adam.c || exit /b 1
call :compile_one lang\dataset_text.c || exit /b 1
call :compile_one lang\dataset_stream.c || exit /b 1
call :compile_one lang\lm_serialize.c || exit /b 1
call :compile_one lang\lm_train.c || exit /b 1
call :compile_one lang\lm_bench.c || exit /b 1
call :compile_one lang\lm_info.c || exit /b 1
call :compile_one lang\quant.c || exit /b 1
call :compile_one lang\kv.c || exit /b 1
call :compile_one cli\tower.c || exit /b 1
call :compile_one cli\chat_session.c || exit /b 1
call :compile_one cli\chat_ui.c || exit /b 1
call :compile_one main.c || exit /b 1
exit /b 0
