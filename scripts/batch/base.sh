#!/bin/bash

ShellScriptFolder=$(cd $(dirname "$0"); pwd)
cd $ShellScriptFolder/../..

set -x

while IFS= read -r scene_id
do
  roslaunch activesplat habitat.launch config:=$3 scene_id:=$scene_id hide_planner_windows:=1 hide_mapper_windows:=1 step_num:=$2 remark:=NONE
done < "$1"