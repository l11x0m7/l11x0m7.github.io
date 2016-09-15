#!/bin/bash

read -p "input the article name with '-' as the interval: " artid

today=`date "+%Y-%m-%d"`

touch _posts/$today-$artid.md
