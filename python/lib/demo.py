import cv2
import sys
import argparse
import glob
from PIL import Image
from Processor import Processor
from Visualizer import Visualizer
import os

def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='/home/valentinasanguineti/Documents/ml_common/yolov5/runs/Torx/Det/torx_T10bw_100epochsfrom100/weights/torx_T10bw.trt', help='trt engine file located in ./models', required=False)
    parser.add_argument('-i', '--image', default='/media/Datasets/Detection_Viti_Torx/imagesnew/IMGS_test/detection_bb/images/img0.png', help='image file path', required=False)
    parser.add_argument('-p', '--folder', default='/media/Datasets/Detection_Viti_Torx/imagesnew/IMGS/T10_bw/detection/images/3', help='folder path', required=False)
    args = parser.parse_args()
    return { 'model': args.model, 'image': args.image, 'folder': args.folder }

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args['model'])
    visualizer = Visualizer()
    path = args['folder']
    for image_name in glob.glob(path + '/*'):#('/home/valentina/Documents/Datasets/centroVitiTest/pos/*'):
        # fetch input
        print('image arg', image_name)
        # image_name = args['image']
        img = cv2.imread('{}'.format(image_name))
        nameimg = image_name.split("/")[-1]
        foldername = "/home/valentinasanguineti/Documents/yolov5-tensorrt/" + image_name.split("/")[-5]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        # inference
        output = processor.detect(img) 
        # img = cv2.resize(img, (640, 640))
        if output!=[]:
            # object visualization
            object_grids = processor.extract_object_grids(output)
            visualizer.draw_object_grid(img, object_grids, 0.1)

            # class visualization
            class_grids = processor.extract_class_grids(output)
            visualizer.draw_class_grid(img, class_grids, 0.01)

            # bounding box visualization
            boxes = processor.extract_boxes(output)
            visualizer.draw_boxes(img, boxes)

            # final results
            boxes, confs, classes = processor.post_process(output)
            visualizer.draw_results(img, boxes, confs, classes, foldername + nameimg)

if __name__ == '__main__':
    main()   
