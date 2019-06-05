import math
import numpy as np
import cv2


class SymbolTracker:

    @staticmethod
    def preprocess_image(image):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_reduced_image = cv2.medianBlur(grayscale_image, 5)
        binary_image = cv2.adaptiveThreshold(noise_reduced_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                             15, 8)
        return binary_image

    @staticmethod
    def detect_components(image):
        connectivity = 8
        components_output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)
        num_labels = components_output[0]
        # The second cell is the label matrix
        labels = components_output[1]
        # The third cell is the stat matrix
        stats = components_output[2]
        # The fourth cell is the centroid matrix
        centroids = components_output[3]
        return labels, stats, centroids

    @staticmethod
    def detect_helipad_candidates(labels, stats, centroids, min_area, max_eccentricity, max_component_distance,
                                  area_ratio):
        euler_numbers = SymbolTracker.get_euler_numbers(labels)
        candidates = range(1, len(stats))
        area_filtered = []
        for candidate in candidates:
            candidate_area = stats[candidate, cv2.CC_STAT_AREA]
            if candidate_area > min_area:
                area_filtered.append(candidate)
        h_candidates = []
        circle_candidates = []
        for candidate in area_filtered:
            if candidate-1 < len(euler_numbers):
                candidate_euler = euler_numbers[candidate-1]
                if candidate_euler == 1:
                    h_candidates.append(candidate)
                elif candidate_euler == 0:
                    candidate_height = stats[candidate, cv2.CC_STAT_HEIGHT]
                    candidate_width = stats[candidate, cv2.CC_STAT_WIDTH]
                    candidate_semimajor_axis = candidate_height if candidate_height > candidate_width else candidate_width
                    candidate_semiminor_axis = candidate_width if candidate_width < candidate_height else candidate_height
                    candidate_eccentricity = math.sqrt(1 - (candidate_semiminor_axis**2 / candidate_semimajor_axis**2))
                    if candidate_eccentricity <= max_eccentricity:
                        circle_candidates.append(candidate)
        filtered_h_candidates = []
        filtered_circle_candidates = []
        for h_candidate in h_candidates:
            for circle_candidate in circle_candidates:
                euclidean_distance = SymbolTracker.get_euclidean_distance(centroids[h_candidate],
                                                                          centroids[circle_candidate])
                if euclidean_distance <= max_component_distance:
                    candidates_area_ratio = stats[h_candidate, cv2.CC_STAT_AREA] / stats[circle_candidate, cv2.CC_STAT_AREA]
                    if abs(candidates_area_ratio - area_ratio) <= 0.05:
                        filtered_h_candidates.append(h_candidate)
                        filtered_circle_candidates.append(circle_candidate)

        return filtered_h_candidates, filtered_circle_candidates

    @staticmethod
    def get_euler_numbers(labels):

        # Adapted from:
        # https://blogs.mathworks.com/steve/2014/10/02/lots-and-lots-of-euler-numbers/
        # Accessed on 5.4.2019.

        lp = np.pad(labels, ((1, 0), (1, 0)), 'constant')

        i_nw = lp[:-1, :-1]
        i_n = lp[:-1, 1:]
        i_w = lp[1:, :-1]

        is_upstream_convexity = np.logical_and(labels, (labels != i_n))
        is_upstream_convexity = np.logical_and(is_upstream_convexity, (labels != i_nw))
        is_upstream_convexity = np.logical_and(is_upstream_convexity, (labels != i_w))

        is_upstream_concavity = np.logical_and(labels, (labels != i_nw))
        is_upstream_concavity = np.logical_and(is_upstream_concavity, (labels == i_n))
        is_upstream_concavity = np.logical_and(is_upstream_concavity, (labels == i_w))

        upstream_convexity_labels = labels[is_upstream_convexity]
        upstream_concavity_labels = labels[is_upstream_concavity]

        total_upstream_convexities = np.bincount(upstream_convexity_labels)[1:]
        # Discard the zero bin, which is the background.
        total_upstream_concavities = np.bincount(upstream_concavity_labels)[1:]

        if len(total_upstream_concavities) > len(total_upstream_convexities):
            total_upstream_concavities = total_upstream_concavities[:len(total_upstream_convexities)]
        elif len(total_upstream_convexities) > len(total_upstream_concavities):
            total_upstream_convexities = total_upstream_convexities[:len(total_upstream_concavities)]

        return total_upstream_convexities - total_upstream_concavities

    @staticmethod
    def get_euclidean_distance(vector1, vector2):
        component_x_distance = (vector1[0] - vector2[0]) ** 2
        component_y_distance = (vector1[1] - vector2[1]) ** 2
        return math.sqrt(component_x_distance + component_y_distance)

    @staticmethod
    def get_painted_components_image(labels):
        # Map component labels to hue val
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0

        return labeled_img

    @staticmethod
    def track_and_paint_helipad(image):
        preprocessed_image = SymbolTracker.preprocess_image(image)
        labels, stats, centroids = SymbolTracker.detect_components(preprocessed_image)
        h_candidates, circle_candidates = SymbolTracker.detect_helipad_candidates(labels, stats, centroids, 1, 0.8, 10,
                                                                                  0.17)
        if len(h_candidates) > 0 and len(circle_candidates) > 0:
            helipad_labels = labels.copy()
            helipad_labels[True] = 0
            for h_candidate in h_candidates:
                h_labels = labels.copy()
                h_labels[h_labels != h_candidate] = 0
                helipad_labels += h_labels
            for circle_candidate in circle_candidates:
                circle_labels = labels.copy()
                circle_labels[circle_labels != circle_candidate] = 0
                helipad_labels += circle_labels
            return SymbolTracker.get_painted_components_image(helipad_labels)
        dummy_image = image.copy()
        dummy_image[True] = 0
        return dummy_image


def capture_webcam_video():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if frame is not None:
            # Our operations on the frame come here
            painted_frame = SymbolTracker.track_and_paint_helipad(frame)

        # Display the resulting frame
            cv2.imshow('painted_frame', painted_frame)
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
