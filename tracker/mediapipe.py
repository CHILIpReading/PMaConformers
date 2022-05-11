""" 
@Author: David Roch
@Date: 10.03.2022
@Description : This Module track all faces in the images
"""

import cv2
import mediapipe as mp
import numpy as np

class FaceDetector():
    def __init__(self, minDetectionConfidence=0.5):
        """ 
        @Input : 
            minDetectionConfidence = Minimume confidence that the detector should have to return the result
        """
        self.minDetectionConfidence = minDetectionConfidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=minDetectionConfidence, min_tracking_confidence=0.5)

    def findFacesMesh(self, img):
        results = self.face_mesh.process(img)

        img_shape = img.shape

        if results.multi_face_landmarks:
            mesh = np.zeros(img_shape)

            for face_landmarks in results.multi_face_landmarks:
                relative_keypoints = {}
                for id, xyz in enumerate(face_landmarks.landmark):
                    x = int(xyz.x*img_shape[1])
                    if x >= img_shape[1]:
                        x = img_shape[1]-1

                    y = int(xyz.y*img_shape[0])
                    if y >= img_shape[0]:
                        y = img_shape[0]-1

                    z = int((minz + xyz.z)*255/(maxz+minz))
                    
        return

