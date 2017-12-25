"""
Arian Niaki
Dec 2017, UMass Amherst
This script is used to detect blockpages using the heuristic length comparison technique
"""


import glob
from bs4 import BeautifulSoup
import math
# f = glob.glob('/scratch/Dropbox/test/block/*.html')
# with open("true.txt",'a') as g:
#     for a in f:
#         g.write(a+'\n')
    
control_path = 'Capture_Path/top500/html/'
import pickle
# control_len = {}
# control_files = glob.glob(control_path+'*.html')
# print control_files
# for fl in control_files:
#     f = open(fl)
#     doc = f.read()
#     f.close()
#     soup = BeautifulSoup(doc, "lxml")
#     for script in soup(['script','style']):
#         script.extract()
#     # body = soup.find('body')
#     body = soup.get_text()
#     if body is not None:
# #                body_text = body
# #                body_text = body.rstrip().lstrip()
#         body_text_temp = body.rstrip().lstrip().replace('\n', '').replace('\t','')
#         # body_text_temp2 = body_text_temp.replace('?',' ')
#         # print len(body_text_temp), fl
#         control_len[fl.split('/')[9]] = len(body_text_temp)

#print control_len['192.html']



def precision_recall_ok(y_true, y_pred):
    num_correctly_detected_blockpages = 0
    detected_blockpage = 0
    true_blockpage = 0
    for i in range(len(y_true)):
        if(y_pred[i]=='ok' and y_true[i]=='ok'):
            num_correctly_detected_blockpages += 1

        if(y_pred[i]=='ok'):
            detected_blockpage += 1
        if(y_true[i]=='ok'):
            true_blockpage += 1
    print("num correctly detected ok ", num_correctly_detected_blockpages)
    print("detected ok ", detected_blockpage)
    print("true ok ", true_blockpage)
    try:
        precision = (num_correctly_detected_blockpages+0.0) / (detected_blockpage+0.0)
        recall = (num_correctly_detected_blockpages+0.0) / (true_blockpage+0.0)
        f1 = 2.0*precision*recall/(precision+recall+0.0)
        print("precision is : %s  recall is : %s , f1 score is for ok: %s" %(str(precision),str(recall),str(f1)))
    except Exception as exp:
        print("exception ",str(exp))

def precision_recall_blockpage(y_true, y_pred):
    num_correctly_detected_blockpages = 0
    detected_blockpage = 0
    true_blockpage = 0
    for i in range(len(y_true)):
        if(y_pred[i]=='block' and y_true[i]=='block'):
            num_correctly_detected_blockpages += 1

        if(y_pred[i]=='block'):
            detected_blockpage += 1
        if(y_true[i]=='block'):
            true_blockpage += 1
    print("num correctly detected blockpages ", num_correctly_detected_blockpages)
    print("detected blockpages ", detected_blockpage)
    print("true blockpages ", true_blockpage)
    try:
        precision = (num_correctly_detected_blockpages+0.0) / (detected_blockpage+0.0)
        recall = (num_correctly_detected_blockpages+0.0) / (true_blockpage+0.0)
        f1 = 2.0*precision*recall/(precision+recall+0.0)
        print("precision is : %s  recall is : %s , f1 score is for BLOCK: %s" %(str(precision),str(recall),str(f1)))

    except Exception as exp:
        print("exception ",str(exp))




import pickle


# with open('control_len.pickle', 'wb') as handle:
#     pickle.dump(control_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

control_len = {}
with open('control_len.pickle', 'rb') as handle:
    control_len = pickle.load(handle)


train_path = 'Dropbox/test/block/'
block_files = glob.glob(train_path+'*.html')
true_block = []

predict_ok = []
predict_block = []

good_block = 0
bad_ok = 0
good_ok = 0
bad_block = 0
print(len(block_files))
for fl in block_files:
    true_block.append(fl.split('/')[5])
    f = open(fl)
    doc = f.read()
    f.close()
    soup = BeautifulSoup(doc, "lxml")
    for script in soup(['script','style']):
        script.extract()
    # body = soup.find('body')
    body = soup.get_text()
    if body is not None:
#                body_text = body
#                body_text = body.rstrip().lstrip()
        body_text_temp = body.rstrip().lstrip().replace('\n', '').replace('\t','')
        # body_text_temp2 = body_text_temp.replace('?',' ')
        import math
        control_length = control_len[fl.split('/')[5].split('_')[1]]
        try:
            diff = abs(float(control_length- len(body_text_temp))*100/max(float(control_length),float(len(body_text_temp))))
            # print diff , fl
            if diff >30.19:
                good_block += 1
                predict_block.append(fl)
            else:
                bad_ok += 1
                predict_ok.append(fl)
        except Exception as exp:
            #print str(exp)
            pass
            # print control_length, len(body_text_temp)

print("NUMBER OF TRUE BLOCK __ BLOCK: ", len(true_block))
print("NUMBER OF DETECTED BLOCK: __ BLOCK", (good_block))
with open('blockdetected.txt','a') as f:
    for i in predict_block:
        f.write(i+'\n')

train_path = 'Dropbox/test/servererr/'
server_files = glob.glob(train_path+'*.html')

for fl in server_files:
    f = open(fl)
    doc = f.read()
    f.close()
    soup = BeautifulSoup(doc, "lxml")
    for script in soup(['script','style']):
        script.extract()
    # body = soup.find('body')
    body = soup.get_text()
    if body is not None:
#                body_text = body
#                body_text = body.rstrip().lstrip()
        body_text_temp = body.rstrip().lstrip().replace('\n', '').replace('\t','')
        # body_text_temp2 = body_text_temp.replace('?',' ')
        import math
        control_length = control_len[fl.split('/')[5].split('_')[1]]
        try:
            diff = abs(float(control_length- len(body_text_temp))*100/max(float(control_length),float(len(body_text_temp))))
            # print diff , fl
            if diff >30.19:
                bad_block += 1
                predict_block.append(fl)
            else:
                bad_ok += 1
                predict_ok.append(fl)
        except Exception as exp:
            pass
            #print str(exp)
            # print control_length, len(body_text_temp)

print("NUMBER OF TRUE BLOCK : __ SERVER ERR", len(true_block))
detected_block = good_block + bad_block
print("NUMBER OF DETECTED BLOCK: __ SERVER ERR", (detected_block))


train_path = 'Dropbox/test/connectionerr/'
connection_files = glob.glob(train_path+'*.html')

for fl in connection_files:
    f = open(fl)
    doc = f.read()
    f.close()
    soup = BeautifulSoup(doc, "lxml")
    for script in soup(['script','style']):
        script.extract()
    # body = soup.find('body')
    body = soup.get_text()
    if body is not None:
#                body_text = body
#                body_text = body.rstrip().lstrip()
        body_text_temp = body.rstrip().lstrip().replace('\n', '').replace('\t','')
        # body_text_temp2 = body_text_temp.replace('?',' ')
        import math
        control_length = control_len[fl.split('/')[5].split('_')[1]]
        try:
            diff = abs(float(control_length- len(body_text_temp))*100/max(float(control_length),float(len(body_text_temp))))
            # print diff , fl
            if diff >30.19:
                bad_block += 1
                predict_block.append(fl)
            else:
                bad_ok += 1
                predict_ok.append(fl)
        except Exception as exp:
            print str(exp)
            # print control_length, len(body_text_temp)


print("NUMBER OF TRUE BLOCK : __ CONNECTION ERR", len(true_block))
detected_block = good_block + bad_block
print("NUMBER OF DETECTED BLOCK: __ CONNECTION ERR", (detected_block))

train_path = 'Dropbox/test/ok/'
ok_files = glob.glob(train_path+'*.html')
true_ok = []

for fl in ok_files:
    f = open(fl)
    doc = f.read()
    f.close()
    soup = BeautifulSoup(doc, "lxml")
    for script in soup(['script','style']):
        script.extract()
    # body = soup.find('body')
    body = soup.get_text()
    if body is not None:
#                body_text = body
#                body_text = body.rstrip().lstrip()
        body_text_temp = body.rstrip().lstrip().replace('\n', '').replace('\t','')
        # body_text_temp2 = body_text_temp.replace('?',' ')
        import math
        control_length = control_len[fl.split('/')[5].split('_')[1]]
        true_ok.append(fl)
        try:
            diff = abs(float(control_length- len(body_text_temp))*100/max(float(control_length),float(len(body_text_temp))))
            # print diff , fl
            if diff >30.19:
                bad_block += 1
                predict_block.append(fl)
            else:
                good_ok += 1
                predict_ok.append(fl)
        except Exception as exp:
            # print str(exp)
            pass
            # print control_length, len(body_text_temp)

print("NUMBER OF TRUE BLOCK : __ OK", len(true_block))
detected_block = good_block + bad_block
print("NUMBER OF DETECTED BLOCK: __OK", (detected_block))
print("BAD BLOCK :L ", bad_block)
print("GOOD OK :L ", good_ok)
print("BAD OK :L ", bad_ok)
print(len(predict_block))
print(len(true_block))
print(len(predict_ok))
print(len(true_ok))
