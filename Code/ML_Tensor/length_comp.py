"""
Arian Niaki
Dec 2017, UMass Amherst
This is for writing all the png and HTML filenames into a text file
"""


import glob
#
# control_path = 'ML_Tensor/control/'
# files = glob.glob(control_path+"*.html")
# a = list(xrange(500))
# controls = []
# for file in files:
#     controls.append(int(file.split('/')[6].replace('.html','')))
# print controls
# for b in a :
#     if b in controls:
#         pass
#     else:
#         print b
#
#     # print fl.split("/")[6].split('_')[1].replace('.html','')


path = 'train/ok/'
ls = glob.glob(path+'*.png')
with open ('png.txt','w') as f:
    for b in ls:
        f.write(b.replace('.png','').replace('train/ok/','')+'\n')



print("__")

path = 'httpme/train/ok/'
gs = glob.glob(path+'*.html')

with open ('html.txt','w') as f:
    for b in gs:
        f.write(b.replace('.html','').replace('httpme/train/ok/','')+'\n')
