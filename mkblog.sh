#!/bin/bash

read -p "input the article name with '-' as the interval: " artid

today=`date "+%Y-%m-%d"`

filename=$today-$artid.md
pathname=_posts/$today-$artid.md
if [[ ! -s $pathname ]];then
    touch $pathname
    echo "--- 
layout: post 
title: 
date: $today 
categories: blog 
tags: [,] 
description: 
--- 
" > $pathname
else
    echo -e "[Error] You've already created the file!"
fi


