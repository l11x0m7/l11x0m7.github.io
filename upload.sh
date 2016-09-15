#! /bin/bash
# -*- encoding:utf8-*-

git add ./
git commit -m "refresh"
git pull origin master
git push -u origin master
