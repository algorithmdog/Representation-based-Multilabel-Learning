#!/bin/bash
rm result/*
bash tool/runonedata.sh enron &
bash tool/runonedata.sh delicious &
bash tool/runonedata.sh eurlex_desc &
