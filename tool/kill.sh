#!/bin/bash
ps aux| grep "run.*sh" |awk '{print $2;}'|xargs kill
ps aux| grep python |awk '{print $2;}'|xargs kill
