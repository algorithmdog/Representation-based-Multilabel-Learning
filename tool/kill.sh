#!/bin/bash
ps aux|grep ll| grep run.sh |awk '{print $2;}'|xargs kill
ps aux|grep ll| grep runonedata.sh | awk '{print $2;}' |xargs kill
ps aux|grep ll| grep python |awk '{print $2;}'|xargs kill
