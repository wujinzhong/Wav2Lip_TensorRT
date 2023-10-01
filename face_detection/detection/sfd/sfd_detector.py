import os
import cv2
from torch.utils.model_zoo import load_url

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *
import torch
import numpy as np

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=os.path.join(os.path.dirname(os.path.abspath(__file__)), 's3fd.pth'), verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if not os.path.isfile(path_to_detector):
            model_weights = load_url(models_urls['s3fd'])
        else:
            model_weights = torch.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()
        self.preload_numpy = torch.from_numpy(np.array([104, 117, 123])).float().to(device)

    def set_trt_engine(self, trt_engine):
        self.trt_engine = trt_engine

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, images, dst_olist=None, events_list=None):
        rng = nvtx.start_range(message="pre_s3fd", color="red")
        images, BB = batch_detect_pre(images, device=self.device, 
                                 preload_numpy = self.preload_numpy)
        nvtx.end_range(rng)

        rng = nvtx.start_range(message="s3fd", color="red")
        olist = batch_detect(self.face_detector, images, device=self.device, trt_engine=self.trt_engine, 
                                 preload_numpy = self.preload_numpy,
                                 dst_olist = dst_olist,
                                 events_list = events_list)
        nvtx.end_range(rng)

        return olist, BB

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
