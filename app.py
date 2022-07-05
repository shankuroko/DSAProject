from flask import Flask, render_template,request,jsonify
import pickle
from werkzeug.utils import secure_filename
import cv2
from matplotlib.pyplot import imshow
import numpy as np
def getImg(imgPath):
  im=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE);
  reImg=cv2.resize(im,(600,800));# 600,800
  return reImg
def getDigits(img):
  marklist={'A':[],'B':[]}
  partB = img[:,187:352] #140
  partA = img[:,60:110] 
  startA=21#24
  incA=20
  startB=20#25
  incB=17
  for i in range(5):
    subStart=0
    incSub=57#61
    th,convert=cv2.threshold(partA[startA:startA+24,:], 118, 250, cv2.THRESH_BINARY) #138
    marklist["A"].append(convert)
    th,markItem=cv2.threshold(partB[startB:startB+22,:], 147, 250, cv2.THRESH_BINARY)
    mark=[]
    for j in range(3):
      mark.append(markItem[:,subStart:subStart+45])#47
      subStart+=incSub
    marklist['B'].append(mark)
    startB+=incB
    startA+=incA
  return marklist
def getFrontPage(img):
  orb=cv2.ORB_create(nfeatures=10000)
  kp1 , des1=orb.detectAndCompute(img,None)
  img2=getImg('./drive/MyDrive/DSA_Lab/mask/mask.jpeg')
  kp2 , des2=orb.detectAndCompute(img2,None)
  bf=cv2.BFMatcher()
  matches=bf.knnMatch(des2,des1,k=2)
  good=[]
  for m,n in matches:
    if m.distance < 0.75*n.distance:
      good.append([m])
  if len(good)>100:
    src_pts = np.float32([ kp2[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img2.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    matrix=cv2.getPerspectiveTransform(dst,pts)
    out= cv2.warpPerspective(img,matrix,(600,800))
    return out
  return None
def printDigits(marks):
  print("Part A:")
  for i in range(5):
    cv2_imshow(marks['A'][i])
    print()
  print("Part B:")
  for j in range(5):
    print(str(6+j))
    for k in range(3):
      cv2_imshow(marks["B"][j][k])
      print()
def getMarksSection(img):
  return img[495:625,35:430]
img=getImg('./drive/MyDrive/DSA_Lab/data/image-045.jpg')
outRes=getFrontPage(img)
finalRes=getMarksSection(outRes)

def getIndividualDigits(marks):
  marksWithDigits={"A":[],"B":[]}
  for i in marks["A"]:
    th,testImg=cv2.threshold(cv2.resize(i,(300,200)),150,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(testImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    numbers=[]
    for i in range(len(sorted_ctrs)):
      x,y,w,h=cv2.boundingRect(sorted_ctrs[i])
      if(w>15 and h>50 and w<185 and h<150):
        d=testImg[y-3:y+h+3,x-3:x+w+3]
        if(len(d)>0):
          numbers.append(d)
    marksWithDigits["A"].append(numbers)
  for i in marks["B"]:
    part=[]
    for j in i:
      th,testImg=cv2.threshold(cv2.resize(j,(300,200)),150,255,cv2.THRESH_BINARY)
      contours,_=cv2.findContours(testImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
      numbers=[]
      for i in range(len(sorted_ctrs)):
        x,y,w,h=cv2.boundingRect(sorted_ctrs[i])
        if(w>15 and h>50 and w<185 and h<150):
          d=testImg[y-3:y+h+3,x-3:x+w+3]
          numbers.append(d)
      if(len(numbers)>0):
        part.append(numbers)
    if(len(part)>0):
      marksWithDigits["B"].append(part)
  return marksWithDigits
marks=getDigits(finalRes)
dMarks=getIndividualDigits(marks)
loaded_model = pickle.load(open('./drive/MyDrive/DSA_Lab/KNN.sav', 'rb'))
loaded_model = pickle.load(open('./drive/MyDrive/DSA_Lab/KNN.sav', 'rb'))
def PredictDigit(img):
  yTest=cv2.resize(img,(28,28)).flatten()
  res=loaded_model.predict([yTest])
  return res[0]
def toCSV(marks):
  mark=[]
  for i in marks["A"]:
    d=''
    for digit in i:
      pred=PredictDigit(digit)
      if(pred<=2 or pred>=10):
        if(pred==11):
          d=d+'.5'
        elif(pred==10):
          d=d+'0'
        else:
          d=d+str(pred)
    mark.append(d)
  for i in marks["B"]:
    for k in i:
      d=''
      for j in k:
        digit=j
        pred=PredictDigit(digit)
        if(pred==11):
          d=d+'.5'
        elif(pred==10):
          d=d+'0'
        else:
          d=d+str(pred)
      mark.append(d)
  return mark
data=toCSV(dMarks)
import csv
f = open('./drive/MyDrive/DSA_Lab/result.csv', 'w')
writer = csv.writer(f)
writer.writerow(data)
f.close()
app= Flask(__name__)

@app.route("/")
def index():
  return render_template("./index.html")
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
  f = request.files['file']
  
  img=secure_filename(f.filename)
  print(img)
  return jsonify(render_template("./index.html",x="Hello"))

if __name__=="__main__":
  app.run(debug=True)