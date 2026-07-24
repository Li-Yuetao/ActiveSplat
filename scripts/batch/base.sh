#!/bin/bash

ShellScriptFolder=$(cd $(dirname "$0"); pwd)
cd $ShellScriptFolder/../..

set -x

while IFS= read -r scene_id
do
  [ -z "$scene_id" ] && continue
  ros2 launch activesplat habitat.launch.py config:="$3" scene_id:="$scene_id" hide_planner_windows:=1 hide_mapper_windows:=1 step_num:="$2" remark:=NONE
  status=$?
  if [ $status -ne 0 ]; then
    echo "ActiveSplat failed for scene $scene_id (exit $status)" >&2
    exit $status
  fi
done < "$1"