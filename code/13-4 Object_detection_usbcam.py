# 导入软件包
import os
import cv2
import numpy as np
import tensorflow as tf
import sys



# 如果当前文件在object_detection文件夹下，那么将上一层路径加入到python搜索路径中
sys.path.append('..')

# 导入工具包
from utils import label_map_util
from utils import visualization_utils as vis_util
# 设置摄像头分辨率
IM_WIDTH = 640    # 使用较小的分辨率，可以得到较快的检测帧率
IM_HEIGHT = 480   

# 使用的模型名字
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# 获取当前工作目录的路径
CWD_PATH = os.getcwd()

# 得到 detect model 文件的路径
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# 得到 label map 文件路径
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# 定义目标检测器检测的目标种类数
NUM_CLASSES = 90

# 加载 label map，并产生检测种类的索引值，以至于，当模型的前向推理计算出预测种类是‘5’，我们
# 能够知道对应的是 ‘飞机’
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 加载 Tensorflow model 到内存中
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# 定义目标检测器的输入，输出张量

# 输入张量是一幅图像
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# 输出张量是检测框，分值以及种类
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# 检测到的目标种类数
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# 初始化帧率
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX


# 初始化USB摄像头
camera = cv2.VideoCapture(0)
ret = camera.set(3,IM_WIDTH)
ret = camera.set(4,IM_HEIGHT)

while(True):

    t1 = cv2.getTickCount()

    # 获取一副图像，并扩展维度成：[1, None, None, 3]
    ret, frame = camera.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # 执行前向检测
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # 画出检测的结果
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.85)

	# 画出帧率
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    
    # 显示图像
    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    # 按 'q' 退出
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()

cv2.destroyAllWindows()
