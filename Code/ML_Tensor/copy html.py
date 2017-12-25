"""
Arian Niaki
Dec 2017, UMass Amherst
This script is used to compare the HTML and IMG files of the webpages
Also to copy all the necessary HTML files to a merged directory of screenshots and HTMLs
"""


import glob

#SRV_DAT_PATH = "train/block/"
#SRV_DAT_PATH = 'train/'
#FILES = glob.glob(SRV_DAT_PATH + "*.png")
#print FILES
#block = []
#for file in FILES:

#    name = file.split('/')[8]
#    block.append(name)
#    if len(name)<10:
#        print name

#for it in block:
#        import subprocess
#
#         # subprocess.check_output(['ls','-l']) #all that is technically needed...
#        command = 'mv'+' /scratch/Dropbox/Dropbox/Rishab/london_http/train/ok/'+it.replace('png','html')+' /scratch/Dropbox/Dropbox/Rishab/london_http/train/block/'
#        subprocess.call(command, shell=True)
#
#        print it,' it'
# #

##
##check directorries
##
PATH = '/london_images/train/ok/'
PATH_HTML = '/london_http/train/ok/'

pngs = glob.glob(PATH+"*.png")
htmls = glob.glob(PATH_HTML+"*.html")

for png in pngs:
    with open ("pngsfiles.txt",'a') as f:
        f.write(png.split('/')[8].replace('.png','')+'\n')

for htm in htmls:
    with open("html0sfiles.txt", 'a') as f:
        f.write(htm.split('/')[8].replace('.html', '') + '\n')


#
# # SRV_DAT_PATH = "train/block/"
# SRV_DAT_PATH = "train/block/"
# FILES = glob.glob(SRV_DAT_PATH + "*.png")
# block = []
# for file in FILES:
#
#     name = file.split('/')[8]
#     block.append(name)
#     if len(name)<10:
#         print name
#
#
# for it in block:
#     if token in it:
#         import subprocess
#
#         # subprocess.check_output(['ls','-l']) #all that is technically needed...
#         command = 'mv'+' /scratch/httpme/w/'+it.replace('png','html')+' /scratch/httpme/block/'
#         subprocess.call(command, shell=True)
#
#         print it, ' it'
