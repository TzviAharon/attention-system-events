import numpy as np
import cv2
from sklearn.cluster import KMeans
import os


class EventDataProcessor:
    """
    Processes event-based data and manages attention mechanisms.
    This class handles loading event data, images, and calibration parameters,
    defines regions of interest (ROI), and visualizes the events with attention zones.
    """

    def __init__(self, event_file, image_file, calib_file, images_folder_path):
        """
        Initializes the EventDataProcessor with event data, images, and calibration.

        Args:
            event_file (str): Path to the event data file.
            image_file (str): Path to the image data file.
            calib_file (str): Path to the calibration data file.
            images_folder_path (str): Path to the frames folder
        """
        self.events = self.load_events(event_file)
        self.images = self.load_images(image_file)
        self.calib = self.load_calibration(calib_file, images_folder_path)
        self.attention_system = AdaptiveAttention(image_size=(self.calib['width'], self.calib['height']))
        self.roi = None

    def load_events(self, filename):
        """
        Loads events from a file.

        Args:
            filename (str): Path to the file containing event data.

        Returns:
            list: A list of events, each represented as a tuple (timestamp, x, y, polarity).
        """
        events = []
        with open(filename, 'r') as f:
            for line in f:
                t, x, y, p = map(float, line.strip().split())
                events.append((t, int(x), int(y), int(p)))
        return events

    def load_images(self, filename):
        """
        Loads image references from a file.

        Args:
            filename (str): Path to the file containing image references.

        Returns:
            dict: A dictionary mapping timestamps to image filenames.
        """
        images = {}
        with open(filename, 'r') as f:
            for line in f:
                t, img_file = line.strip().split()
                images[float(t)] = img_file
        return images

    def load_calibration(self, calib_filename, images_filename):
        """
        Loads calibration parameters from a file.

        Args:
            calib_filename (str): Path to the calibration file.
            images_filename (str): Path to the frame images directory

        Returns:
            dict: A dictionary containing camera calibration parameters and image dimensions.
        """
        with open(calib_filename, 'r') as f:
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = map(float, f.read().strip().split())

        files = os.listdir(images_filename)
        first_file_path = os.path.join(images_filename, files[0])
        img = cv2.imread(first_file_path)
        return {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3,
            'width': img.shape[0], 'height': img.shape[1]
        }

    def define_roi(self):
        """
        Allows the user to define a region of interest (ROI) on a blank image.
        The ROI is set in the attention system.
        """
        img = np.zeros((self.calib['width'], self.calib['height'], 3), dtype=np.uint8)
        roi = cv2.selectROI("Define ROI", img)
        cv2.destroyWindow("Define ROI")
        self.roi = roi
        self.attention_system.set_roi(roi)

    def process_events_in_window(self, start_time, end_time, attention_mode):
        """
        Processes events within a specified time window.

        Args:
            start_time (float): Start time of the window.
            end_time (float): End time of the window.
            attention_mode (str): Mode of attention ('size', 'roi', 'memory').

        Returns:
            tuple: A tuple containing the window events and the calculated attention zone, or None if no events are found.
        """
        window_events = [e for e in self.events if start_time <= e[0] < end_time]
        if not window_events:
            return None

        event_array = np.array([(e[1], e[2]) for e in window_events])
        attention_zone = self.attention_system.process_events(event_array, attention_mode)
        return window_events, attention_zone

    def visualize_events_and_attention(self, attention_zone, image_file):
        """
        Visualizes events and the attention zone on an image.

        Args:
            attention_zone (tuple): Coordinates of the attention zone.
            image_file (str): Path to the image file.
        """
        img = cv2.imread(image_file)

        # Draw the attention zone
        if attention_zone is not None:
            x, y = map(int, attention_zone)
            # cv2.drawMarker(img, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2.circle(img, (x, y), 30, (0, 0, 255))

        # Draw the region of interest (ROI) if defined
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)


        cv2.imshow('Events and Attention', img)
        cv2.waitKey(1)


class AdaptiveAttention:
    """
    Adaptive attention system that processes events and determines areas of interest.
    Uses K-Means clustering to identify clusters of events and applies different attention modes.
    """

    def __init__(self, image_size, n_clusters=3):
        """
        Initializes the AdaptiveAttention system.

        Args:
            image_size (tuple): Size of the image (width, height).
            n_clusters (int): Number of clusters for K-Means.
        """
        self.image_size = image_size
        self.n_clusters = n_clusters
        self.previous_attention = None
        self.roi = None

    def set_roi(self, roi):
        """
        Sets the region of interest (ROI) for the attention system.

        Args:
            roi Sequence[int]: Coordinates of the ROI (x, y, width, height).
        """
        self.roi = roi

    def process_events(self, events, attention_mode):
        """
        Processes a list of events and determines the attention zone.

        Args:
            events (np.array): Array of event coordinates (x, y).
            attention_mode (str): Mode of attention ('size', 'roi', 'memory').

        Returns:
            np.array: Coordinates of the attention zone or None if not enough events are present.
        """
        if len(events) < self.n_clusters:
            return None

        # Apply K-Means clustering to the events
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(events)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)

        # Determine the attention mode
        if attention_mode == 'size':
            weights = cluster_sizes
        elif attention_mode == 'roi':
            weights = self.roi_weighting(centroids, cluster_sizes)
        elif attention_mode == 'memory':
            if self.previous_attention is None:
                weights = cluster_sizes
            else:
                weights = self.memory_weighting(centroids)
        else:
            weights = cluster_sizes

        # Select the cluster with the highest weight as the attention zone
        attention_index = np.argmax(weights)
        attention_zone = centroids[attention_index]

        self.previous_attention = attention_zone
        return attention_zone

    def roi_weighting(self, centroids, cluster_sizes):
        """
        Applies weighting to clusters based on their proximity to the ROI.

        Args:
            centroids (np.array): Centroid coordinates of the clusters.
            cluster_sizes (np.array): Sizes of the clusters.

        Returns:
            np.array: Adjusted weights for each cluster.
        """
        if self.roi is None:
            return cluster_sizes

        x, y, w, h = self.roi
        weights = np.zeros_like(cluster_sizes, dtype=float)
        for i, (cx, cy) in enumerate(centroids):
            if x <= cx <= x + w and y <= cy <= y + h:
                weights[i] = cluster_sizes[i]
            else:
                weights[i] = cluster_sizes[i] * 0.05  # Reduced importance for clusters outside ROI
        return weights

    def memory_weighting(self, centroids):
        """
        Applies memory-based weighting to clusters based on their distance from the previous attention zone.

        Args:
            centroids (np.array): Centroid coordinates of the clusters.

        Returns:
            np.array: Adjusted weights for each cluster.
        """
        distances = np.linalg.norm(centroids - self.previous_attention, axis=1)
        weights = np.exp(-distances / 50)  # Gaussian-like weighting
        return weights


def main():
    """
    Main function to process event data, apply attention mechanisms, and visualize results.
    This function loads the necessary data, allows the user to select the attention mode,
    processes events in a sliding window, and visualizes the events and attention zones.
    """

    processor = EventDataProcessor('events.txt', 'images.txt', 'calib.txt', 'images')


    attention_mode = ''
    while attention_mode not in ['size', 'roi', 'memory']:
        attention_mode = input("Choose attention mode (size/roi/memory): ").lower()

    if attention_mode == 'roi':
        # Ask user to define ROI
        processor.define_roi()

    window_size = 0.03  # 50ms window
    start_time = processor.events[0][0]
    end_time = processor.events[-1][0]
    current_time = start_time
    while current_time <= end_time:
        window_end = current_time + window_size
        result = processor.process_events_in_window(current_time, window_end, attention_mode)

        if result:
            window_events, attention_zone = result
            nearest_image_time = min(processor.images.keys(), key=lambda x: abs(x - current_time))
            image_file = processor.images[nearest_image_time]
            processor.visualize_events_and_attention(attention_zone, image_file)

        current_time = window_end

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
