#!/usr/bin/env python3

import os.path

directories = ["words/", "words/a02/a02-000/"]
files = ["words.txt", "test.png", "words/a02/a02-000/a02-000-00-00.png"]

for i in directories:
    if os.path.isdir(i):
        print('OK', i)
    else:
        print('ERROR', i)

for i in files:
    if os.path.isfile(i):
        print("OK", i)
    else:
        print("ERR", i)
