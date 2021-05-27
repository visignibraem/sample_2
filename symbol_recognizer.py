import logging

import numpy as np

from lpr.exporter import Exporter
from lpr.model import BaseModel
from lib.utils_cv2 import filter_zero_boxes


class SymbolRecognizer(BaseModel):
    def additional_init(self):
        logging.info('{} is running'.format(self.__class__.__name__))
        self.exporter = Exporter()

    def decode_prediction(self):
        results = self.result_q['results'][:3]
        all_scores = results[1]
        # This is the event we must add info to
        event = self.result_q['extras']['event']
        # If we have all scores <=0, we can not handle it further (no symbols at all)
        if all_scores.any():
            event.sm_boxes, event.sm_scores, event.sm_names, event.sm_words = [], [], [], []
            # for each imput image we have a separate element in boxes, scores and classes
            for boxes, scores, classes in zip(*results[:3]):
                try:
                    # Each of the filters returns idx to reconstruct array. They should be applied one by one.
                    boxes, idx = np.unique(boxes[scores>0], axis=0, return_index=True)
                    scores, classes = scores[idx], classes[idx]
                    boxes, idx = filter_zero_boxes(boxes)
                    scores, classes = scores[idx], classes[idx]
                except ValueError:
                    # no symbols at all
                    boxes, scores, classes = [], [], []
                # Note: here we may have no symbols at all, but we can't drop it
                # e.g. first plate -> no symbols, second plate -> 9 symbols
                sm_names = [self.category_index[x]['name'] for x in classes]
                if sm_names:
                    # To make text representation, we sort symbols by x_min coord of its box
                    sorted_by_x = sorted(zip(sm_names, boxes[:, 1]), key=lambda p: p[1])
                    text = ''.join([p[0].upper() for p in sorted_by_x])
                    event.sm_words.append(text)

                # Note: we use `scores`, not `all_scores` here.
                event.sm_boxes.append(boxes)
                event.sm_scores.append(scores)
                event.sm_names.append(sm_names)

            logging.debug('Symbols found: {}'.format(event.sm_names))
            logging.debug('Plates found: {}'.format(event.sm_words))

        # After we saved all information in the Event, we mark it as passed the stage.
        # If we didn't save any, we still marked it as passed the stage.
        event.stage_done()
        return event

    def get_event(self):
        return self.qe.get()

    def get_input(self, evt):
        try:
            return evt.plates
        except AttributeError:
            # if it has no plates we consider it empty
            evt.stage_done()
            self.exporter.put(evt, fps=self.qe.mps)

    def export(self):
        event = self.decode_prediction()
        # logging.debug('Symbols found before filter: {}'.format(len(decoded_result[4][3])))
        self.exporter.put(event, fps=self.qe.mps)
