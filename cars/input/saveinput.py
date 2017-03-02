from glob import glob

fil = open("input.csv",'w')
imagefiles = glob("*.jpg")
for i,im in enumerate(imagefiles):
    if i<len(imagefiles)-1:
        fil.write(im+"\n")
    else:
        fil.write(im)
fil.close()