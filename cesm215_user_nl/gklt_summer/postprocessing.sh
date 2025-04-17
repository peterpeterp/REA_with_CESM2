#!/bin/bash

source ~/cesm215_scripts/settings.sh

CASE=$(pwd | awk '{split($0,a,"/"); print a[length(a)]}')

echo $ARCHIVE/$CASE

echo "$(date)" >> $ARCHIVE/$CASE/postprocessed_this.txt