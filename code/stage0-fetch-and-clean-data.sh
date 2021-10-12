#!/bin/sh -e 

mkdir -p temp ; cd temp

curl -LO https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip

unzip student.zip

tr -d'"' < student-por.csv | tr ';' ',' > student-por-clean.csv
