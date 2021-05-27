import logging
import time
import itertools

import numpy as np
import os

from lib.utils_cv2 import get_rois, filter_zero_boxes, gray_to_rgb
from lib.video_reader import VideoReader
from lpr.model import BaseModel
from lib.event import Event


class PlateDetector(BaseModel):
    def additional_init(self):
        logging.info('{} is running'.format(self.__class__.__name__))
        # We have precomputed this when loading, but try to compute here if not (compatibility).
        try:
            self.has_camera = self.cfg['has_camera']
        except KeyError:
            self.has_camera = '{camera' in self.cfg['pipeline']

        try:
         logging.info('Plugins: {}'.format(os.environ['GST_PLUGIN_PATH']))
         logging.info('Display: {}'.format(os.environ['DISPLAY']))
        except KeyError:
            logging.info('Not all of $DISPLAY and $GST_PLUGIN_PATH is set.')
        self.video_readers = []
        if self.has_camera:
            # we start a VideoReader for each camera
            for cam in self.cfg['cameras']:
                self.video_readers.append(VideoReader(cam=cam))
        else:
            # we have only one VideoReader for the pipeline
            # but have to pass the default and a fake id in
            cam = {'id': 'video'}
            cam.update(self.cfg['camera_defaults'])
            self.video_readers.append(VideoReader(cam=cam))

        # start each of the readers
        l = len(self.video_readers)
        logging.info('Creating {} readers ({})'.format(l, self.cfg['cameras'] if self.has_camera else 'video'))
        for vr in self.video_readers:
            vr.bind(self.cfg['pipeline'], self.cfg['width'], self.cfg['height'], self.cfg.get('type', None)).start()

        # construct the iterator (reader, id)
        self.video_reader = zip(itertools.cycle(self.video_readers),
                                itertools.cycle([r.cam['id'] for r in self.video_readers] if self.has_camera else '0')
                                )

    def get_event(self):
        # for now read from each reader in round-robbin manner (1 frame from each)
        reader, cam = next(self.video_reader)
        frame = reader.read()
        if frame is None:
            # the line below was commented to avoid spam in debug logs. See #74
            # logging.info('\x1b[6;30;42mPlate detector receive None from video_reader (cam_id:{})\x1b[0m'.format(cam))
            return None
        # if we have 2dim image, we assume this is grayscale. If this is 3dim and not rgb, we still don't know what
        # to do with it so will raise.
        frame = gray_to_rgb(frame) if frame.ndim == 2 else frame
        # Push the new Event (will be used in `self.start`)
        event = Event()
        event.frame = frame
        event.time = time.time()
        event.cam_id = cam
        logging.debug('Plate detector received image from camera {}'.format(cam))
        event.stage_done()
        return event

    def get_input(self, evt):
        # Return a list of one image
        return np.expand_dims(evt.frame, axis=0)

    def export(self):
        # Note: the code below works under the assumption that input is given to the net one by one
        # See `symbol_recognizer` for batch version
        boxes, scores, classes = self.result_q['results'][0], self.result_q['results'][1], self.result_q['results'][2]
        boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)

        event = self.result_q['extras']['event']

        boxes, idx = filter_zero_boxes(boxes)
        scores, classes = scores[idx], classes[idx]

        rois = get_rois(event.frame, boxes, scores,
                        max_boxes=20, min_score_thresh=self.cfg['det_th'],
                        coords_norm=True, expand=self.cfg['expand_method'])

        th_scores = scores[scores > self.cfg['det_th']]
        th_boxes = boxes[scores > self.cfg['det_th']]

        logging.debug('Image shape:{}, cam_id:{}, plate found:{}'.format(event.frame.shape, event.cam_id, len(rois)))
        if len(rois) == len(th_scores) and len(rois) > 0:
            event.plates, event.pl_boxes, event.pl_scores = rois, th_boxes, th_scores
        event.stage_done()
        # Now we are pulling an event in any case. Has it some plates or not? Time will tell.
        self.qe.put(event)

    def decode_prediction(self):
        raise NotImplementedError("This class doesn't have this method")
