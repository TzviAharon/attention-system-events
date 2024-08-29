import numpy as np
import cv2
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from collections import defaultdict


class EventDataProcessor:
    def __init__(self, event_file, image_file, calib_file):
        self.events = self.load_events(event_file)
        self.images = self.load_images(image_file)
        self.calib = self.load_calibration(calib_file)
        self.attention_system = AdaptiveAttention(image_size=(self.calib['width'], self.calib['height']))

    def load_events(self, filename):
        events = []
        with open(filename, 'r') as f:
            for line in f:
                t, x, y, p = map(float, line.strip().split())
                events.append((t, int(x), int(y), int(p)))
        return events

    def load_images(self, filename):
        images = {}
        with open(filename, 'r') as f:
            for line in f:
                t, img_file = line.strip().split()
                images[float(t)] = img_file
        return images

    def load_calibration(self, filename):
        with open(filename, 'r') as f:
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = map(float, f.read().strip().split())
        return {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3,
            'width': int(cx * 2), 'height': int(cy * 2)  # Estimate image size
        }

    def process_events_in_window(self, start_time, end_time):
        window_events = [e for e in self.events if start_time <= e[0] < end_time] # obtain all events in the time window
        if not window_events: # no event in the window
            return None

        event_array = np.array([(e[1], e[2]) for e in window_events])  # obtain the coordinates of the events
        attention_zone = self.attention_system.process_events(event_array)
        return window_events, attention_zone

    def visualize_events_and_attention(self, window_events, attention_zone, image_file):
        img = cv2.imread(image_file)
        for _, x, y, p in window_events:
            color = (0, 0, 255) if p == 1 else (255, 0, 0)
            cv2.circle(img, (x, y), 1, color, -1)

        if attention_zone is not None:
            x, y = map(int, attention_zone)
            cv2.drawMarker(img, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        cv2.imshow('Events and Attention', img)
        cv2.waitKey(0)


class AdaptiveAttention:

    def __init__(self, image_size, n_clusters=3):
        self.image_size = image_size
        self.n_clusters = n_clusters
        self.previous_attention = None

    def process_events(self, events, weighting_scheme='size'):  # events variable is ndarray of the coordinates of the events

        if len(events) < self.n_clusters:
            return None

        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(events)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)

        if weighting_scheme == 'size':
            weights = cluster_sizes
        elif weighting_scheme == 'memory':
            weights = self.memory_weighting(centroids)
        else:
            weights = cluster_sizes

        attention_index = np.argmax(weights) # get index of the most valuable centroid
        attention_zone = centroids[attention_index]

        self.previous_attention = attention_zone
        return attention_zone

    def memory_weighting(self, centroids):
        if self.previous_attention is None:
            return np.ones(len(centroids))
        distances = np.linalg.norm(centroids - self.previous_attention, axis=1)
        weights = np.exp(-distances / 50)
        return weights


def main():
    processor = EventDataProcessor('events.txt', 'images.txt', 'calib.txt')

    window_size = 50/1000  # 50ms window
    start_time = processor.events[0][0]
    end_time = processor.events[-1][0]

    current_time = start_time
    while current_time < end_time:
        window_end = current_time + window_size
        result = processor.process_events_in_window(current_time, window_end)  # events in windows, region of interest

        if result:
            window_events, attention_zone = result
            nearest_image_time = min(processor.images.keys(), key=lambda x: abs(x - current_time))
            image_file = processor.images[nearest_image_time].split('/')[1]
            image_file = os.path.join(os.getcwd(), 'slide_hdr_close', 'images', image_file)
            processor.visualize_events_and_attention(window_events, attention_zone, image_file)

        current_time = window_end

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()