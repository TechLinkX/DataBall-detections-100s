#-*-coding:utf-8-*-
# date:2024-09
# Author: DataBall Xian
# function: show yolo data of voc format anno

import cv2
import os
import numpy as np
import xml.etree.cElementTree as et
import supervision as sv

if __name__ == "__main__":

    path_data='D:/wjx/dataset/img_list/bike/'

    idx = 0
    box_annotator = sv.BoxAnnotator()

    for file in os.listdir(path_data):
        if ".jpg" in file or ".png" in file:
            print(" ->[{}] {}".format(idx,file))
            path_img = path_data + file
            path_label = path_img.replace(".jpg",".xml").replace(".png",".xml")
            if not os.access(path_label,os.F_OK): # 判断标注文件是否存在
                continue
            img = cv2.imread(path_img) # 读取图片

            tree=et.parse(path_label)
            root=tree.getroot()
            for Object in root.findall('object'):
                name=Object.find('name').text # 获取类别名字
                # 获取坐标 xyxy
                bndbox=Object.find('bndbox')
                x1= np.float32((bndbox.find('xmin').text))
                y1= np.float32((bndbox.find('ymin').text))
                x2= np.float32((bndbox.find('xmax').text))
                y2= np.float32((bndbox.find('ymax').text))

                # opencv 方式可视化
                # cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (255,100,100), 2)
                # cv2.putText(img, "{}".format(name), (int(x1),int(y1)),\
                # cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 55, 255), 6)
                # cv2.putText(img, "{}".format(name), (int(x1),int(y1)),\
                # cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2)

                # sv.BoxAnnotator() 方式可视化
                box_ = np.array([int(x1),int(y1), int(x2),int(y2)]).reshape(-1,4)
                det_ = sv.Detections(xyxy=box_)
                img = box_annotator.annotate(scene=img, detections=det_, labels=[name])

            cv2.namedWindow('image',0)
            cv2.imshow('image',img)
            if cv2.waitKey(30) == 27:
                break
    cv2.destroyAllWindows()
