#!/bin/bash
wget -c -q --show-progress -i mnist.list 
ls *.gz | xargs -I% gunzip -k %
