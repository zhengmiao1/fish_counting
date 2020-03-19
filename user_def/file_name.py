import os
dir1 = "/root/data/fish_data1/fish2019.9.16/"
def ListFilesToTxt(dir,file,wildcard,recursion):
    exts = wildcard.split(",")
    files = os.listdir(dir)
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    file.write(dir1+name + "\n")
                    break
def Test():
  dir = "E:\Postgraduate\data"
  outfile="fish.txt"
  wildcard =".jpg"

  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)
  ListFilesToTxt(dir,file,wildcard, 1)

  file.close()
Test()