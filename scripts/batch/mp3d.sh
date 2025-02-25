#!/bin/bash

ShellScriptFolder=$(cd $(dirname "$0"); pwd)
cd $ShellScriptFolder/../..
WorkspaceFolder=$(pwd)
echo $WorkspaceFolder

set -x

$ShellScriptFolder/base.sh $1 $2 $WorkspaceFolder/config/datasets/mp3d.json