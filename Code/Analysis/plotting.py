import matplotlib.pyplot as plt
## plotting for blockpage
model0001_128_block=[ 0.35396039604, 0,  0.765407554672, 0.863445378151, 0.968122786305, 0.968122786305 , 0.966981132075,  0.966981132075,0.976635514019 , 0.977829638273 , 0.983758700696,0.990783410138 , 0.988479262673 ]
epoch_model0001_128_block = [1,5,10,20,30,40,50,53,58,70,80,92,99]

html0001_128_block=[0,0.746438746439,0.746438746439, 0.965761511216,0.966981132075, 0.965761511216, 0.980,0.980,0.964622641509]
epoch_html0001_128_block = [1,5,10,20,30,40,50,60,66]




png0001_256_block = [0,0,0,0.785,0.892,0.9436,0.9532,0.9756,0.9830,0.9820,0.9838,0.990825688073,0.990825688073,0.990825688073]
epoch_png0001_256_block=[1,5,10,20,30,40,50,60,70,80,90,99,115,122]

fig = plt.figure()

plt.plot(epoch_png0001_256_block,png0001_256_block,label=r'IMAGE $\alpha=0.0001$ fully-connected(2)')
plt.plot(epoch_html0001_128_block,html0001_128_block,label=r'HTML $\alpha=0.0001$ fully-connected(1)')
plt.plot(epoch_model0001_128_block,model0001_128_block,label=r'Model $\alpha=0.0001$ fully-connected(1)')
plt.xlim(0,120)
plt.ylim(0,1)


plt.plot
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.legend()
plt.show()





html001_256_ok=[0.910116681626, 0.990508096036, 0.990925589837, 0.990925589837, 0.990925589837,0.990925589837,0.990925589837,0.990925589837,0.990925589837,0.990925589837,0.990925589837,0.990925589837]
epoch_html001_256_ok = [1,5,10,20,35,40,50,60,80,90,95,102]


model0001_128_ok= [0.914016172507, 0.910116681626, 0.981742738589, 0.989958158996,0.991063948618 , 0.997319034853,  0.998308429659, 0.998591152437, 0.998309859155 ,0.998733286418 ,0.998733286418 ,0.998873873874 ,0.999154929577]

epoch_model0001_128_ok = [1,5,10,20,30,40,53,58,70,80,90,92,99]

png0001_256_ok = [0.88275499474,0.910116681626, 0.910116681626,  0.973980798664 , 0.983337961677,0.994938132733, 0.99634214969, 0.997045998031 , 0.997045166737,  0.997326579429,0.997466929355, 0.998168756163, 0.998309859155, 0.998027613412,0.998309859155]
epoch_png0001_256_ok=[1,5,10,20,30,40,50,60,70,80,90,99,110,115,122]

fig = plt.figure()

plt.plot(epoch_png0001_256_ok,png0001_256_ok,label=r'IMAGE $\alpha=0.0001$ fully-connected(2)')
plt.plot(epoch_html001_256_ok,html001_256_ok,label=r'HTML $\alpha=0.001$ fully-connected(2)')
plt.plot(epoch_model0001_128_ok,model0001_128_ok,label=r'Model $\alpha=0.0001$ fully-connected(1)')
plt.legend(loc=4)
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.xlim(0,120)
plt.ylim(0,1)

plt.show()



html001_256_server=[0,0,0,0.96682464455,0.96682464455 ,0.96682464455 ,0.96682464455 ,  0.971428571429,0.971428571429, 0.971428571429,0.971428571429 ,0.971428571429  ]
epoch_html001_256_server = [1,5,10,20,35,40,50,60,80,90,95,102]

model001_128_server= [0,0,0.96682464455,0.96682464455,0.96682464455,0.961904761905,0.971428571429, 0.971428571429,0.971428571429, 0.971428571429,0.971428571429,0.971428571429]
epoch_model001_128_server = [1,5,10,20,30,40,49,57,64,82,87,94]


png001_128_server = [0,0,0,0.9,0.969,0.972,0.971,0.976076555024, 0.976076555024,0.971428571429,0.971428571429,0.971428571429,0.976076555024]
epoch_png001_128_server=[1,5,10,20,30,40,50,60,70,80,90,92,99]

fig = plt.figure()

plt.plot(epoch_png001_128_server,png001_128_server,label=r'IMAGE $\alpha=0.001$ fully-connected(1)')
plt.plot(epoch_html001_256_server,html001_256_server,label=r'HTML $\alpha=0.001$ fully-connected(2)')
plt.plot(epoch_model001_128_server,model001_128_server,label=r'Model $\alpha=0.001$ fully-connected(1)')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.xlim(0,120)
plt.ylim(0,1)

plt.show()




html001_256_connection =[0,0,0.826568265683,0.826568265683 , 0.860215053763,0.860215053763 ,0.860215053763, 0.860215053763, 0.86021505376]
epoch_html001_256_connection = [1,5,10,20,30,40,50,59,66]


png001_256_connection = [0,0,0,0,0.941558441558, 0.962025316456,0.984126984127 , 0.957928802589 ,0.986175115207, 0.964169381107, 0.987261146497 ,0.937888198758]
epoch_png001_256_connection=[1,5,10,20,30,40,50,60,70,80,85,92]


model0001_128_connection= [0,0 ,0 ,0.826568265683 , 0.860215053763,0.881355932203 , 0.901734104046,0.912280701754 ,0.930930930931 ,0.948012232416 ,0.959752321981 , 0.974842767296,0.965732087227]

epoch_model0001_128_connection = [1,5,10,20,30,40,53,58,70,80,90,92,99]



fig = plt.figure()

plt.plot(epoch_png001_256_connection,png001_256_connection,label=r'IMAGE $\alpha=0.001$ fully-connected(1)')
plt.plot(epoch_html001_256_connection,html001_256_connection,label=r'HTML $\alpha=0.001$ fully-connected(2)')
plt.plot(epoch_model0001_128_connection,model0001_128_connection,label=r'Model $\alpha=0.0001$ fully-connected(1)')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.xlim(0,120)
plt.ylim(0,1)

plt.show()