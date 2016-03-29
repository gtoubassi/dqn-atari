#!/bin/bash

echo -n "Average score first 50:  "
grep Episode $1 | head -50 | awk '{print $6}' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{print SUM/COUNT}'

echo -n "Average score last 50:   "
grep Episode $1 | tail -50 | awk '{print $6}' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{print SUM/COUNT}'

echo -n "Average score first 100: "
grep Episode $1 | head -100 | awk '{print $6}' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{print SUM/COUNT}'

echo -n "Average score last 100:  "
grep Episode $1 | tail -100 | awk '{print $6}' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{print SUM/COUNT}'

echo -n "Average score first 250: "
grep Episode $1 | head -250 | awk '{print $6}' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{print SUM/COUNT}'

echo -n "Average score last 250:  "
grep Episode $1 | tail -250 | awk '{print $6}' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{print SUM/COUNT}'

echo -n "Average score first 500: "
grep Episode $1 | head -500 | awk '{print $6}' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{print SUM/COUNT}'

echo -n "Average score last 500:  "
grep Episode $1 | tail -500 | awk '{print $6}' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{print SUM/COUNT}'

echo -n "Average score for ALL:  "
grep Episode $1 | awk '{print $6}' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{print SUM/COUNT}'

echo
echo "Best scores ever:"
grep Episode $1 | awk '{print $6}' | sort -n | tail -5 

