import os, random,shutil
data_base_dir0 = "./test2/CNN_synth_testset/0_real" 
data_base_dir1 = "./test2/CNN_synth_testset/1_fake" 
tarDirTrain0 = "test2/CNN_synth_testset/train/0_real" 
tarDirTrain1 ="test2/CNN_synth_testset/train/1_fake" 
tarDirTest0 = "test2/CNN_synth_testset/test/0_real" 
tarDirTest1 = "test2/CNN_synth_testset/test/1_fake" 
count0 = 0 #记录训练数据集图像搬移的数量
count1 =0	#记录测试集图像搬移的数量


#获取每个文件夹下的500张图片存在新的路径下
def moveFileTrain(img_dir_path,tarDirTrain ):
    pathDir = os.listdir(img_dir_path) # 取图片的原始路径
    sample = random.sample(pathDir,500)	#随机选取500数量的样本图片	
    global count0
    for name in sample: 
        shutil.move(img_dir_path + '/'+ name, tarDirTrain +'/'+ name)
        #os.rename(tarDirTrain +'/'+ name , tarDirTrain +'/’ + str(count0) + ".png") 
        count0 += 1 
    return
#搬移测试集图像
def moveFileTest(img_dir_path,tarDirTest):
    pathDir = os.listdir(img_dir_path) 
    filenumber =len(pathDir)
    sample = random.sample(pathDir,filenumber) 
    global count1
    for name in sample:
        shutil.move(img_dir_path + '/' + name, tarDirTest +'/'+ name)
        # os.rename(tarDirTest +'/'+ name , tarDirTest +'/’ + str(count1) + ".png") 
        count1 += 1 
    return


moveFileTrain(data_base_dir0,tarDirTrain0) 
moveFileTrain(data_base_dir1,tarDirTrain1) 
moveFileTest(data_base_dir0,tarDirTest0) 
moveFileTest(data_base_dir1,tarDirTest1)

# pytorch封装好的划分数据集函数，torch.utils.data.random_split()也可以完成对数据的随机划分。