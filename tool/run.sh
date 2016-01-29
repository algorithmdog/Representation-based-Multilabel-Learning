#!/bin/bash
#bash tool/run_leml_cv.sh enron 25 
##bash tool/run_leml_cv.sh enron 50
#bash tool/run_leml_cv.sh delicious 250 
#bash tool/run_leml_cv.sh delicious 500
#bash tool/run_leml_cv.sh eurlex_desc  250
#bash tool/run_leml_cv.sh eurlex_desc 500
#bash tool/run_leml.sh lshtc 250
#bash tool/run_leml.sh lshtc 500

#bash tool/run_rep_cv.sh enron 25
#bash tool/run_rep_cv.sh enron 50
##bash tool/run_rep_cv.sh delicious 250
#bash tool/run_rep_cv.sh delicious 500
#bash tool/run_rep_cv.sh eurlex_desc 250
#bash tool/run_rep_cv.sh eurlex_desc 500
#bash tool/run_rep.sh lshtc 250
#bash tool/run_rep.sh lshtc 500

bash tool/run_wsabie_cv.sh enron 25 &
bash tool/run_wsabie_cv.sh enron 50 &
bash tool/run_wsabie_cv.sh delicious 250 &
bash tool/run_wsabie_cv.sh delicious 500 &
bash tool/run_wsabie_cv.sh eurlex_desc 250 &
bash tool/run_wsabie_cv.sh eurlex_desc 500 &
bash tool/run_wsabie.sh lshtc 250 &
bash tool/run_wsabie.sh lshtc 500 &
