from tracker.face_tracker import FaceTracker

def main():
    retina_fan = FaceTracker(device="cpu", model="retina_fan")
    mediapipe = FaceTracker(device="cpu", model="mediapipe")

if __name__ == "__main__" :
    main()