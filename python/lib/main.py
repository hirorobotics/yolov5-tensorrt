import cv2
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import os
import glob

def draw_transparent_frame(image, thickness, alpha=0.5, color=(0, 0, 0)):
    """
    Draws a semi-transparent frame with the specified thickness on an image.

    Args:
        image: The input image (NumPy array).
        thickness: Thickness of the frame in pixels.
        alpha: Transparency level (0.0 to 1.0).
        color: Color of the frame (BGR tuple).
    """
    overlay = image.copy()
    h, w, _ = image.shape

    # Draw the four lines of the frame
    cv2.rectangle(overlay, (0, 0), (w, thickness), color, -1)  # Top line
    cv2.rectangle(overlay, (0, h - thickness), (w, h), color, -1)  # Bottom line
    cv2.rectangle(overlay, (0, thickness), (thickness, h - thickness), color, -1)  # Left line
    cv2.rectangle(overlay, (w - thickness, thickness), (w, h - thickness), color, -1)  # Right line

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

class YOLOv5TRT:
    def __init__(self, engine_path, input_shape=(1600, 1216)):  # Width, Height
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.TRT_LOGGER)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape
        self.inputs, self.outputs, self.bindings = self.allocate_buffers()
        self.stream = cuda.Stream()

        # Postprocessing parameters (adjust as needed)
        self.conf_thres = 0.1
        self.iou_thres = 0.6
        self.num_classes = 1  # Single class detection

    def load_engine(self, engine_path):
        with open(engine_path, 'rb') as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if (self.engine.get_tensor_mode(binding).name == 'INPUT'):
                inputs.append({'host': host_mem, 'device': device_mem})
            elif (self.engine.get_tensor_mode(binding).name == 'OUTPUT'):
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings

    def preprocess_image(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_shape)
        img = img.transpose((2, 0, 1)).astype(np.float16)
        img /= 255.0
        return np.ascontiguousarray(img)

    def infer(self, image):
        img_preprocessed = self.preprocess_image(image)
        # print("preprocessing {}".format(img_preprocessed.dtype))
        np.copyto(self.inputs[0]['host'], img_preprocessed.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        self.context.execute_v2(bindings=self.bindings)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()
        # print("Raw output shapes:", [out['host'].shape for out in self.outputs])
        # print("Raw output values (first few):", self.outputs[0]['host'][:10]) #or other output index
        return [out['host'] for out in self.outputs]

    def postprocess_detections(self, outputs):
        detections = np.concatenate([out.reshape(1, -1, self.num_classes + 5) for out in outputs], axis=1)[0].astype(np.float32)

        boxes = detections[:, :4]
        confs = detections[:, 4]
        classes = detections[:, 5:]

        # Apply confidence threshold
        keep = confs > self.conf_thres
        boxes = boxes[keep]
        confs = confs[keep]
        classes = classes[keep]

        # Convert xywh to xyxy
        if boxes.size > 0:
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

            # NMS (Non-Maximum Suppression)
            keep = self.nms(boxes_xyxy, confs)
            boxes_xyxy = boxes_xyxy[keep]
            confs = confs[keep]
            classes = classes[keep]

            return boxes_xyxy, confs, classes
        else:
            return np.array([]), np.array([]), np.array([])

    def nms(self, boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.iou_thres)[0]
            order = order[inds + 1]
        return keep

    def detect(self, image):
        outputs = self.infer(image)
        return self.postprocess_detections(outputs)

if __name__ == "__main__":
    # export LD_LIBRARY_PATH=/home/valentinasanguineti/Downloads/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-11.8/TensorRT-10.7.0.23/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
    #./trtexec --fp16 --onnx=/home/valentinasanguineti/Documents/ml_common/yolov5/runs/Torx/Det/Mixed8mm/torx_mixed8mm_and_12mm/weights/8mmand12mm.onnx --saveEngine=/home/valentinasanguineti/Documents/ml_common/yolov5/runs/Torx/Det/Mixed8mm/torx_mixed8mm_and_12mm/weights/8mmand12mm.trt
    engine_path = "/home/valentinasanguineti/Documents/ml_common/yolov5/runs/Torx/Det/Mixed8mm/torx_mixed8mm_and_12mm/weights/8mmand12mm.trt"  # Replace with your engine file path
    #image_path = "/media/Datasets/Detection_Viti_Torx/imagesnew/IMGS_test/detection_bb/images/test/1/img1.png"  # Replace with your image path
    path = "/media/Datasets/Detection_Viti_Torx/images/T10_bw/1/"
    yolo_trt = YOLOv5TRT(engine_path)
    for image_path in glob.glob(path + '/*'):#('/home/valentina/Documents/Datasets/centroVitiTest/pos/*'):
        # fetch input
        # print('image ', image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (1600,1216))
        nameimg = image_path.split("/")[-1]
        foldername = "/home/valentinasanguineti/Documents/yolov5-tensorrt/" + image_path.split("/")[-5]
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        if image is None:
            print(f"Error: Could not read image from {image_path}")
        else:
            boxes, confs, classes = yolo_trt.detect(image)

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Draw a transparent frame
            frame_thickness = 250  # Adjust thickness
            transparency = 0.3  # Adjust transparency (0.0 to 1.0)
            frame_color = (255, 0, 0) #blue

            draw_transparent_frame(image, frame_thickness, transparency, frame_color)

            # cv2.imshow("Detected Objects", image)
            success = cv2.imwrite('{}'.format(foldername + '/' + nameimg), image)