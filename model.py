import time

import numpy as np
import tensorflow as tf
import logging
from lib.labelmap_util import load_label_map
from lib.load_graph_nms_v2 import LoadFrozenGraph
from lib.session_worker import SessionWorker

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class BaseModel:
    def __init__(self, cfg, roi_input_queue):
        self.qe = roi_input_queue
        # TODO: get rid of these two
        self.queue_data = None
        self.result_q = None
        # self.last_evt = None
        self.cfg = cfg
        self.category_index = load_label_map(cfg)
        self.sleep_interval = 0.005
        self.tensor_shape = (300, 300, 3)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = cfg['allow_memory_growth']
        config.gpu_options.force_gpu_compatible = cfg['force_gpu_compatible']
        config.gpu_options.per_process_gpu_memory_fraction = cfg['gpu_memory_fraction']
        graph = LoadFrozenGraph(self.cfg).load_graph()
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.gpu_worker = SessionWorker('GPU', graph, config)
        self.gpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]

        self._dummy_run() # Speedup!
        self.additional_init()

        self.box_count = 0
        self.results = []

    def _dummy_run(self):
        gpu_feeds = {self.image_tensor: [np.zeros(self.tensor_shape)]}
        gpu_extras = {}
        self.gpu_worker.put_sess_queue(self.gpu_opts, gpu_feeds, gpu_extras)

        while True:  # Waiting JIT
            g = self.gpu_worker.get_result_queue()
            if g is None:
                time.sleep(0.1)
            else:
                break

    def additional_init(self):
        pass

    def decode_prediction(self):
        raise NotImplementedError("Please Implement this method")

    def get_event(self):
        raise NotImplementedError("Please Implement this method")

    def get_input(self, evt):
        raise NotImplementedError("Please Implement this method")

    def export(self):
        raise NotImplementedError("Please Implement this method")

    def start(self, save=True):
        while True:
            # TODO: work with exceptions
            if not self.gpu_worker.is_result_empty():  # Always export events from result_queue if not empty!
                self.result_q = self.gpu_worker.get_result_queue()  # max_size=1, after that result_q is strictly empty
                self.export()

            # For now, we handle one event at a time. Firstly, get this `Event`
            input_event = self.get_event()
            if input_event is None:
                # no input this time, see you later
                continue
            # ... and then, get images from it, either one or many (a list or ndarray).
            input_images = self.get_input(input_event)
            if input_images is None:
                # None is the marker that this event has no data to process
                continue

            if not isinstance(input_images, (list, np.ndarray)):
                time.sleep(self.sleep_interval)  # TODO: is it necessary?
                continue

            gpu_feeds = {self.image_tensor: input_images}
            gpu_extras = dict()

            assert input_event.cam_id is not None, 'Camera id is None! Did you forget to push it?'
            # an Event is now the only thing we pass through
            gpu_extras['event'] = input_event
            # put new elem in worker's input; might block. maxsize=1, after that sess_queue is full and blocked.
            self.gpu_worker.put_sess_queue(self.gpu_opts, gpu_feeds, gpu_extras)

