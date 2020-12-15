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



def upSize(path,save_path,size):
    dirs = os.path.join(path)
    img_list = os.listdir(dirs)
    img_list.sort(key=lambda x: int(x[:-9]))
    ind = 0
    for i in img_list:
        img_array = cv2.imread(os.path.join(dirs, i), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (size, size), interpolation=cv2.INTER_NEAREST)
        temp = save_path
        save_path = save_path + "/" + str(ind) + '.png'
        ind = ind + 1
        cv2.imwrite(save_path, new_array)
        save_path = temp


def cutImg(path,save_path,size):
    path2 = os.path.join(path)
    img_list = os.listdir(path2)
    ind = 0

    for i in img_list:
        img_array = cv2.imread(os.path.join(path2, i), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (size, size), interpolation=cv2.INTER_NEAREST)
        # gray = cv2.cvtColor(new_array, cv2.COLOR_RGB2GRAY)
        # new_array = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
        save_pathOrigion = save_path
        save_path = save_path + "/" + str(ind) + '.png'
        ind = ind + 1
        cv2.imwrite(save_path, new_array)
        save_path = save_pathOrigion


def firstTrain(phaseName,size,parameter):
    cleanModel()
    copyFiles('../AutoRun/cut/' + str(size) + 'x/A', '../AutoRun/' + phaseName + '/combinedForRun/A/train')
    copyFiles('../AutoRun/cut/' + str(size) + 'x/B', '../AutoRun/' + phaseName + '/combinedForRun/B/train')
    os.system(
        'python ../datasets/combine_A_and_B.py --fold_A ../AutoRun/' + phaseName + '/combinedForRun/A --fold_B ../AutoRun/' + phaseName + '/combinedForRun/B --fold_AB ../AutoRun/' + phaseName + '/combinedForRun')
    os.system('python ../train.py --dataroot ../AutoRun/' + phaseName + '/combinedForRun --name phase' + str(size) + ' --load_size ' + str(size) + ' --crop_size ' + str(size) + ' ' + parameter)


def restOfTrain(size,trainParameter,testParameter,cutSize,isLast):
    if not isLast:
        nextSize = cutSize[cutSize.index(size)+1]
        publicPath = '../AutoRun/phase' + str(size)
        #shutil.copytree('../AutoRun/checkpoints/phase'+ str(size), '../AutoRun/phase' + str(size) + '/model')
        copyFiles('../AutoRun/checkpoints', '../AutoRun/phase' + str(size) + '/model')
        #shutil.rmtree('../AutoRun/checkpoints')
        os.system('python ../test.py --dataroot ../AutoRun/phase' + str(size)+ '/combinedForRun/A/train --name phase' + str(size) + ' --load_size ' + str(size) + ' --crop_size ' + str(size) + ' --checkpoints_dir ../AutoRun/phase' + str(size) + '/model --results_dir ../AutoRun/phase' + str(size) + '/test '+ testParameter)
        os.mkdir(publicPath + '/test/testForAdd_F')
        os.mkdir(publicPath + '/test/testForAdd_R')
        split(publicPath + '/test/testForAdd_F',publicPath + '/test/testForAdd_R',publicPath + '/test/phase' + str(size) + '/test_latest/images')
        add(publicPath + '/test/testForAdd_F', publicPath + '/combinedForRun/A/train', publicPath + '/added')
        upSize(publicPath + '/test/testForAdd_F', publicPath + '/upSized',int(nextSize))
        cleanModel()
        copyFiles(publicPath + '/upSized', '../AutoRun/phase' + str(nextSize) + '/combinedForRun/A/train')
        copyFiles('../AutoRun/cut/' + str(nextSize) + 'x/B', '../AutoRun/phase' + str(nextSize) + '/combinedForRun/B/train')
        os.system(
            'python ../datasets/combine_A_and_B.py --fold_A ../AutoRun/phase' + str(nextSize) + '/combinedForRun/A --fold_B ../AutoRun/phase' + str(nextSize) + '/combinedForRun/B --fold_AB ../AutoRun/phase' + str(nextSize) + '/combinedForRun')
        os.system(
            'python ../train.py --dataroot ../AutoRun/phase' + str(nextSize) + '/combinedForRun --name phase' + str(nextSize) + ' --load_size '+ str(nextSize) +' --crop_size ' + str(nextSize) + ' ' + trainParameter)
    else:
        publicPath = '../AutoRun/phase' + str(size)
        # shutil.copytree('../AutoRun/checkpoints/phase'+ str(size), '../AutoRun/phase' + str(size) + '/model')
        copyFiles('../AutoRun/checkpoints', '../AutoRun/phase' + str(size) + '/model')
        # shutil.rmtree('../AutoRun/checkpoints')
        os.system('python ../test.py --dataroot ../AutoRun/cut/' + str(size) + 'x/A --name phase' + str(
            size) + ' --load_size ' + str(size) + ' --crop_size ' + str(
            size) + ' --checkpoints_dir ../AutoRun/phase' + str(size) + '/model --results_dir ../AutoRun/phase' + str(
            size) + '/test ' + testParameter)
        os.mkdir(publicPath + '/test/testForAdd_F')
        os.mkdir(publicPath + '/test/testForAdd_R')
        split(publicPath + '/test/testForAdd_F', publicPath + '/test/testForAdd_R',
              publicPath + '/test/phase' + str(size) + '/test_latest/images')

#----------------------------------------------------------------------------------------------------------------
sourceImg = "/home/qhf/pytorch-CycleGAN-and-pix2pix-master/train/256x"
trainParameter = "--model pix2pix --netG FCN  --direction AtoB --dataset_mode aligned --norm batch"
testParameter = "--model test --netG FCN  --direction AtoB --dataset_mode single --norm batch --num_test 600"
cutSize = [25,32,64,128,256]
#----------------------------------------------------------------------------------------------------------------

for i in cutSize:
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
for i in cutSize:
    times = str(i) + 'x'
    if not os.path.exists('../AutoRun/cut/' + times):
        os.mkdir('../AutoRun/cut/' + times)
    if not os.path.exists('../AutoRun/cut/' + times + '/A'):
        os.mkdir('../AutoRun/cut/' + times + '/A')
    if not os.path.exists('../AutoRun/cut/' + times + '/B'):
        os.mkdir('../AutoRun/cut/' + times + '/B')
    if os.listdir('../AutoRun/cut/' + times + '/A'):
        shutil.rmtree('../AutoRun/cut/' + times + '/A')
        os.mkdir('../AutoRun/cut/' + times + '/A')
    if os.listdir('../AutoRun/cut/' + times + '/B'):
        shutil.rmtree('../AutoRun/cut/' + times + '/B')
        os.mkdir('../AutoRun/cut/' + times + '/B')
for i in cutSize:
    times = str(i) + 'x'
    cutImg(sourceImg + '/A', '../AutoRun/cut/' + times + '/A',i)
    cutImg(sourceImg + '/B', '../AutoRun/cut/' + times + '/B', i)
firstTrain('phase' + str(cutSize[0]), cutSize[0],trainParameter)
isLast = False
for i in cutSize:
    if i == cutSize[len(cutSize)-1] :
        isLast = True
    restOfTrain(i,trainParameter,testParameter,cutSize,isLast)

