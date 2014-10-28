import numpy as np
import matplotlib.pyplot as plt

caffe_root = '/home/cygnus/caffe-contrastive-loss/caffe-contrastive-loss/'
import sys
sys.path.append(caffe_root + 'python')
#sys.path.append('/usr/lib/python2.7/dist-packages/')

import caffe
import cv2

net = caffe.Classifier(caffe_root + 'examples/imagenet/imagenet_deploy.prototxt',
                       caffe_root + 'examples/imagenet/caffe_reference_imagenet_model')
net.set_phase_test()
net.set_mode_cpu()
net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
#net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
cap = cv2.VideoCapture(1)

while True:
    try:    
        ret, frame = cap.read()        
        if not ret:
            raise IOException
    except:
        print "Camera not ready..."
        continue
    cv2.imshow("frame" ,frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
    if key == 13: 
        #frame = cv2.equalizeHist(frame)
        caffe_input = cv2.resize(frame, (256, 256)).astype(np.float32)
        #caffe_input = cv2.cvtColor(caffe_input, cv2.COLOR_RGB2BGR)

        scores = net.predict([caffe_input])
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

        # sort top k predictions from softmax output
        ans = net.blobs['prob'].data[4].flatten()
        top_k = ans.argsort()[-1:-6:-1]
        prob = ans[top_k]
        for i in range(5):
            print prob[i], labels[top_k[i]]
        print '\n'
