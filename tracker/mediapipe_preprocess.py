""" 
@Author: David Roch
@Date: 10.03.2022
@Description : This Module track all faces in the images
"""

import cv2
import mediapipe as mp
import json
import numpy as np

class Mediapipe():
    def __init__(self, minDetectionConfidence=0.5):
        """ 
        @Input : 
            minDetectionConfidence = Minimume confidence that the detector should have to return the result
        """
        self.minDetectionConfidence = minDetectionConfidence

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=minDetectionConfidence,
                                                     min_tracking_confidence=0.5)

        with open("mediapipe_mapping/test", "r") as fp:
            self.map = json.load(fp)

        self.bbox = None
        self.landmarks = None

    def convertToPMaFormat(self, face_landmarks):
        bbox = [0,0,float("inf"),float("inf")]
        for face_landmark in face_landmarks:
            if face_landmark[0]<bbox[0]:
                bbox[0] = face_landmark[0]
            elif face_landmark[1]<bbox[1]:
                bbox[1] = face_landmark[1]
            elif face_landmark[0]>bbox[2]:
                bbox[2] = face_landmark[0]
            elif face_landmark[1]>bbox[3]:
                bbox[3] = face_landmark[1]
        return bbox

    def get_landmarks_and_score(self):
        return self.landmarks, None

    def findFacesMesh(self, img):
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img_shape = img.shape
        f_landmarks = []

        if results.multi_face_landmarks:
            for xyz in results.multi_face_landmarks[0].landmark:
                x = int(xyz.x*img_shape[1])
                if x >= img_shape[1]:
                    x = img_shape[1]-1

                y = int(xyz.y*img_shape[0])
                if y >= img_shape[0]:
                    y = img_shape[0]-1
                f_landmarks.append((x,y))
        
        f_landmarks = np.array([f_landmarks[int(i)] for i in self.map])

        return f_landmarks

    def findLandmarks(self, img):
        face_landmarks = self.findFacesMesh(img)
        bbox = self.convertToPMaFormat(face_landmarks)
        self.bbox = bbox
        self.landmarks = face_landmarks
        return self.bbox

    def __call__(self, frame, rgb):
        return self.findLandmarks(frame)
