""" 
@Author: David Roch, Daniel Tozadore
@Date: 01.05.2022
@Description: 
    Comparison of Retina Fan landmarks with medipipe landmarks and map the Mediapipe points to the closest
    Retina FAN one using a single image or the first image of a video.
"""

from tracker.face_tracker import FaceTracker

import numpy as np
import math
import statistics
import cv2


image_path = "/home/david/test_landmarks.jpg"



def compute_distance_map(retina_landmarks, mediapipe_landmarks):
    map = np.zeros((len(retina_landmarks),len(mediapipe_landmarks)))
    for r_id, r_landmark in enumerate(retina_landmarks):
        for m_id, m_landmark in enumerate(mediapipe_landmarks):
            map[r_id, m_id] = math.sqrt((r_landmark[0]-m_landmark[0])**2+(r_landmark[1]-m_landmark[1])**2)
    return map

def compute_closest_landmarks(distance_map):
    map = np.zeros(distance_map.shape[0])
    min_dists = []
    for i in range(distance_map.shape[0]):
        min_dist = float("inf")
        for j in range(distance_map.shape[1]):
            if min_dist > distance_map[i,j]:
                min_dist = distance_map[i,j]
                map[i] = j
        min_dists.append(min_dist)
    return map, min_dists

def main():
    retina_fan = FaceTracker(device="cpu", model="retina_fan")
    mediapipe = FaceTracker(device="cpu", model="mediapipe", comparison = True)

    retina_landmarks = retina_fan.tracker(image_path)
    mediapipe_landmarks = mediapipe.tracker(image_path)

    print("retina_landmarks shape: ", len(retina_landmarks), len(retina_landmarks[0]), len(retina_landmarks[0][0]))
    print("mediapipe_landmarks shape: ", len(mediapipe_landmarks), len(mediapipe_landmarks[0]), len(mediapipe_landmarks[0][0]))

    distances_maps = []
    distances_map = compute_distance_map(retina_landmarks[0], mediapipe_landmarks[0])
    distances_maps.append(distances_map)
    map, min_dists = compute_closest_landmarks(distances_maps[0])
    print(statistics.mean(min_dists))


    import json
    with open("mediapipe_mapping/test", "w") as fp:
        json.dump(list(map), fp)
    
    with open("mediapipe_mapping/test", "r") as fp:
        map = np.array(json.load(fp))

    img = cv2.imread(image_path)

    retina_img = img.copy()
    for l in retina_landmarks[0]:
        cv2.circle(retina_img, (int(l[0]), int(l[1])), radius=2, color=(0, 0, 255), thickness=-1)
    #cv2.imshow("retina_img",retina_img)

    mediapipe_img = retina_img.copy()
    for l in mediapipe_landmarks[0]:
        cv2.circle(mediapipe_img, l, radius=1, color=(255, 0, 0), thickness=-1)
    #cv2.imshow("mediapipe_img",mediapipe_img)

    map_img = mediapipe_img.copy()
    for p in map:
        cv2.circle(map_img, mediapipe_landmarks[0][int(p)], radius=1, color=(0, 255, 0), thickness=-1)
    cv2.imshow("map_img",map_img)

    while True:
        cv2.waitKey(1)


if __name__ == "__main__" :
    main()