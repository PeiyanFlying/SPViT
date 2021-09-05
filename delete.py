import os
from time import sleep
import argparse

root = "./"

f_list = []
for dirpath, dirs, files in os.walk(root):
    for filename in files:
        fname = os.path.join(dirpath,filename)
        if ".DS_Store" in fname:
                f_list.append(fname)

for f in f_list:
    print(f)

input("Press any key to continue. File will be removed eternally.")

for dirpath, dirs, files in os.walk(root):
    for filename in files:
        fname = os.path.join(dirpath,filename)
        if fname in f_list:
            print("To remove:", fname)
            os.system("rm {}".format(fname))