import os
import numpy as np
import argparse
from PIL import Image
import onnxruntime as ort
from scipy.spatial.distance import cdist

# ext2 = ('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG')

def extract_feat(sess, filename, input_size):
    img = Image.open(filename)
    img = img.resize((input_size, input_size))
    img  = np.array(img)
    img = img.astype(np.float32)
    img = img/255
    
    img[:,:,0] = (img[:,:,0] - 0.485)/0.229
    img[:,:,1] = (img[:,:,1] - 0.456)/0.224
    img[:,:,2] = (img[:,:,2] - 0.406)/0.225
    
    img = img.transpose((2, 0, 1))
    # print(img.shape)
    im = img[np.newaxis, :, :, :]
    
    # im = im.astype(np.float32)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    result = sess.run(None, {input_name: im})
    # print(result)
    oup = result[0]
    return oup

def load_onnx_model(modelpath):
   
    ort.set_default_logger_severity(3)
    so = ort.SessionOptions()

    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1

    EP_list = ['CPUExecutionProvider']
    sess = ort.InferenceSession(modelpath, providers=EP_list, sess_options=so)

    return sess

def main(args):
    
    anchor_image = args.anchor_path
    probe_image = args.probe_path
    threshold = args.threshold
    input_size = args.input_size
    sess = load_onnx_model(args.model_path)
     
    anchor_feat = extract_feat(sess, anchor_image, input_size)
    probe_feat  = extract_feat(sess, probe_image, input_size)

    pscore = 1 - cdist(anchor_feat.reshape(1, -1), probe_feat.reshape(1, -1), 'cosine').item()
    
    print('Similarity score:{}, threshold: {}'.format(pscore, threshold))
    

    if pscore >= threshold:
        print('The model thinks the images are thesame')
    else:
        print('The model thinks the images aren\'t thesame')
    
def parse_args():
    description = \
    '''
    This script can be used to check if two images are the same or not 

    Usage:
    python3 image-similarity-check.py 
        python3 image-similarity-check.py -a /fullpath/to/anchor-image -c /fullpath/to/probe-image -m /fullpath/to/onnx-model
   
    
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-a', '--anchor_path', action='store', help='absolute path to the anchor image', required=True)
    parser.add_argument('-c', '--probe_path', action='store', help='absolute path to the probe image', required=True)
    parser.add_argument('-t', '--threshold', action='store', help='threshold between 0 and 1', default=0.8,  type=float, required=False)
    parser.add_argument('-m', '--model_path', action='store', help='absolute path to the onnx model', default='similarity-model-512input.onnx', required=False)
    parser.add_argument('-s', '--input_size', action='store', help='input size expected by the model',  default=512, type=int, required=False)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
