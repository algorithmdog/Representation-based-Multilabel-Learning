#!/bin/bash
rm result/*
bash tool/runonedata.sh enron 25 &
bash tool/runonedata.sh delicious 200  &
bash tool/runonedata.sh eurlex_desc 600 
bash tool/runonedata.sh lshtc 200 

