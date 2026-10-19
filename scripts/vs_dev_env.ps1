# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

# Dot-source to load the x64 Visual Studio C++ build environment into this process.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $vsPath) { throw 'Visual Studio C++ build tools were not found' }
$vsDevCmd = Join-Path $vsPath 'Common7\Tools\VsDevCmd.bat'
$envLines = & $env:ComSpec /d /s /c "`"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 && set"
if ($LASTEXITCODE -ne 0) { throw 'VsDevCmd failed to initialize the compiler environment' }
foreach ($line in $envLines) {
    $equals = $line.IndexOf('=')
    if ($equals -gt 0) { [Environment]::SetEnvironmentVariable($line.Substring(0, $equals), $line.Substring($equals + 1), 'Process') }
}
