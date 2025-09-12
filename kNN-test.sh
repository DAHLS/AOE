#!/bin/bash
for i in {100000..6174656..500000}
do
    python AOE_BoW-Extended.py --algo kNN --feat $i
done
