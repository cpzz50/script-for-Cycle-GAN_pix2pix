import os
import shutil
import cv2
from PIL import Image
import numpy as np


def copyFiles(sourceDir, targetDir):
    if sourceDir.find("exceptionfolder") > 0:
        return
    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir, file)
        targetFile = os.path.join(targetDir, file)
        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) !=
                                                                                  os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
        if os.path.isdir(sourceFile):
            copyFiles(sourceFile, targetFile)


def cleanModel():
    if os.listdir('../checkpoints'):
        shutil.rmtree('../checkpoints')
        os.mkdir('../checkpoints')


def add(targetA, targetB, path):
    img_list = os.listdir(os.path.join(targetA))
    img_list.sort(key = lambda x: int(x[:-9]))
    img_list2 = os.listdir(os.path.join(targetB))
    img_list2.sort(key = lambda x: int(x[:-4]))
    en = 0
    for file in img_list:
        save_path = path + '/' + str(en) + '.png'
        file2 = img_list2[en]
        im1 = Image.open(targetA + '/' + file)
        width, height = im1.size
        data = im1.getdata()
        data = np.matrix(data, dtype='float') / 255
        mt1 = np.reshape(data.getA(), (height, width, 3))
        im2 = Image.open(targetB + '/' + file2)
        width, height = im2.size
        data = im2.getdata()
        data = np.matrix(data, dtype='float') / 255
        # 这里处理的是rgb三通道的图片
        mt2 = np.reshape(data.getA(), (height, width, 3))
        mt3 = ((mt1 + mt2) / 2)*255
        img_now = Image.fromarray(mt3.astype(np.uint8))
        img_now.save(save_path)
        en += 1


def split(a, b, path):
    img_list = os.listdir(os.path.join(path))
    img_list.sort(key=lambda x: int(x[:-9]))
    for file in img_list:
        img = path + '/' + file
        if file.endswith('fake.png'):
            shutil.copy(img, a)
        else:
            shutil.copy(img, b)



def upSize(path,save_path,size,size2):
    dirs = os.path.join(path)
    img_list = os.listdir(dirs)
    img_list.sort(key=lambda x: int(x[:-4]))
    ind = 0
    for i in img_list:
        img_array = cv2.imread(os.path.join(dirs, i), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (size2, size), interpolation=cv2.INTER_NEAREST)
        temp = save_path
        save_path = save_path + "/" + str(ind) + '.png'
        ind = ind + 1
        cv2.imwrite(save_path, new_array)
        save_path = temp


def cutImg(path,save_path,size,size2):
    path2 = os.path.join(path)
    img_list = os.listdir(path2)
    img_list.sort(key=lambda x: int(x[1:-5]))
    ind = 0

    for i in img_list:
        img_array = cv2.imread(os.path.join(path2, i), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (size2, size), interpolation=cv2.INTER_NEAREST)
        # gray = cv2.cvtColor(new_array, cv2.COLOR_RGB2GRAY)
        # new_array = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
        save_pathOrigion = save_path
        save_path = save_path + "/" + str(ind) + '.png'
        ind = ind + 1
        cv2.imwrite(save_path, new_array)
        save_path = save_pathOrigion




def firstTest(i,testParameter,cutSize2,v,e):
    publicPath = '../AutoRun/phase' + str(i)
    dataroot = '--dataroot ../AutoRun/cut/' + str(i) + 'x/A'
    checkPionts_dir = '--checkpoints_dir ../checkpoints '
    results_dir = ' --results_dir ../AutoRun/phase' + str(i) + '/test '
    os.system('python ../test.py ' + dataroot + ' --name phase' + str(i) + ' --load_size1 ' + str(i) + ' --load_size2 '+ str(cutSize2) + ' --crop_size ' + str(cutSize2) + ' ' + checkPionts_dir + results_dir + testParameter)
    os.mkdir(publicPath + '/test/testForAdd_F')
    os.mkdir(publicPath + '/test/testForAdd_R')
    split(publicPath + '/test/testForAdd_F', publicPath + '/test/testForAdd_R',
          publicPath + '/test/phase' + str(i) + '/test_latest/images')
    add(publicPath + '/test/testForAdd_F', '../AutoRun/cut/' + str(i) + 'x/A', publicPath + '/added')
    upSize(publicPath + '/added', publicPath + '/upSized', int(e),int(v))

def testInit(i,testParameter,cutSize,isLast,cutSize2):
    dataroot = '--dataroot ../AutoRun/phase' + str(i) + '/upSized'
    checkPionts_dir = '--checkpoints_dir ../checkpoints'
    if not isLast:
        thisSize = cutSize[cutSize.index(i) + 1]
        nextSize = cutSize[cutSize.index(i) + 2]
        results_dir = ' --results_dir ../AutoRun/phase' + str(thisSize) + '/test '
        publicPath = '../AutoRun/phase' + str(thisSize)
        os.mkdir('../AutoRun/phase' + str(thisSize) + '/test/testForAdd_R')
        os.mkdir('../AutoRun/phase' + str(thisSize) + '/test/testForAdd_F')
        os.system('python ../test.py ' + dataroot + ' --name phase' + str(thisSize) + ' --load_size1 ' + str(thisSize) + ' --load_size2 '+ str(cutSize2[cutSize1.index(thisSize)]) + ' --crop_size ' + str(cutSize2[cutSize1.index(thisSize)]) + ' ' + checkPionts_dir + results_dir + testParameter)
        split(publicPath + '/test/testForAdd_F', publicPath + '/test/testForAdd_R',
              publicPath + '/test/phase' + str(thisSize) + '/test_latest/images')
        add(publicPath + '/test/testForAdd_F', '../AutoRun/phase' + str(i) + '/upSized', publicPath + '/added')
        upSize(publicPath + '/added', publicPath + '/upSized', int(nextSize),cutSize2[cutSize.index(nextSize)])
    else:
        os.system('python ../test.py --dataroot ../AutoRun/phase' + str(cutSize[-2]) + '/upSized --name phase' + str(cutSize[-1]) + ' --load_size1 ' + str(
            cutSize[-1]) + ' --load_size2 ' + str(cutSize2[-1]) + ' --crop_size ' + str(cutSize2[-1]) + ' ' + checkPionts_dir + ' --results_dir ../AutoRun/phase' + str(cutSize[-1]) + '/test ' + testParameter)
        os.mkdir('../AutoRun/phase' + str(cutSize[cutSize.index(i) + 1]) + '/test/testForAdd_R')
        os.mkdir('../AutoRun/phase' + str(cutSize[cutSize.index(i) + 1]) + '/test/testForAdd_F')
        split('../AutoRun/phase' + str(cutSize[cutSize.index(i) + 1]) + '/test/testForAdd_F', '../AutoRun/phase' + str(cutSize[cutSize.index(i) + 1]) + '/test/testForAdd_R',
              '../AutoRun/phase' + str(cutSize[cutSize.index(i) + 1]) + '/test/phase' + str(str(cutSize[cutSize.index(i) + 1])) + '/test_latest/images')








sourceImg = "/home/qhf/GAN/fingerResult/tested/Input"
testParameter = "--model test --netG FCN  --direction AtoB --dataset_mode single --norm batch --num_test 1269"
cutSize1 = [25,43,61,79,97]
cutSize2 = [94,161,229,296,364]

for i in cutSize1:
    phase = 'phase' + str(i)
    if not os.path.exists(phase):
        os.mkdir(phase)
    if not os.path.exists('../AutoRun/' + phase + '/added'):
        os.mkdir('../AutoRun/' + phase + '/added')
    if not os.path.exists('../AutoRun/' + phase + '/combinedForRun'):
        os.mkdir('../AutoRun/' + phase + '/combinedForRun')
    if not os.path.exists('../AutoRun/' + phase + '/model'):
        os.mkdir('../AutoRun/' + phase + '/model')
    if not os.path.exists('../AutoRun/' + phase + '/test'):
        os.mkdir('../AutoRun/' + phase + '/test')
    if not os.path.exists('../AutoRun/' + phase + '/upSized'):
        os.mkdir('../AutoRun/' + phase + '/upSized')
    if not os.path.exists('../AutoRun/' + phase + '/combinedForRun/A'):
        os.mkdir('../AutoRun/' + phase + '/combinedForRun/A')
    if not os.path.exists('../AutoRun/' + phase + '/combinedForRun/A/train'):
        os.mkdir('../AutoRun/' + phase + '/combinedForRun/A/train')
    if not os.path.exists('../AutoRun/' + phase + '/combinedForRun/B'):
        os.mkdir('../AutoRun/' + phase + '/combinedForRun/B')
    if not os.path.exists('../AutoRun/' + phase + '/combinedForRun/B/train'):
        os.mkdir('../AutoRun/' + phase + '/combinedForRun/B/train')
    if not os.path.exists('../AutoRun/' + phase + '/combinedForRun/train'):
        os.mkdir('../AutoRun/' + phase + '/combinedForRun/train')
if not os.path.exists('../AutoRun/cut'):
    os.mkdir('../AutoRun/cut')
for i in cutSize1:
    times = str(i) + 'x'
    if not os.path.exists('../AutoRun/cut/' + times):
        os.mkdir('../AutoRun/cut/' + times)
    if not os.path.exists('../AutoRun/cut/' + times + '/A'):
        os.mkdir('../AutoRun/cut/' + times + '/A')
    if os.listdir('../AutoRun/cut/' + times + '/A'):
        shutil.rmtree('../AutoRun/cut/' + times + '/A')
        os.mkdir('../AutoRun/cut/' + times + '/A')
for i in cutSize1:
    times = str(i) + 'x'
    cutImg(sourceImg, '../AutoRun/cut/' + times + '/A',i,cutSize2[cutSize1.index(i)])
firstTest(cutSize1[0],testParameter,cutSize2[0],cutSize2[1],cutSize1[1])
isLast = False
for i in cutSize1:
    if i == cutSize1[len(cutSize1)-2] :
        isLast = True
    testInit(i,testParameter,cutSize1,isLast,cutSize2)