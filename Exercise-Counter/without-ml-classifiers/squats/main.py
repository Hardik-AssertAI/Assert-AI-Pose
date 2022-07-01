import cv2
import sys
from counters.using3DAngles import squatCounter

def main():
    counter = squatCounter()

    cap = cv2.VideoCapture(sys.argv[1])

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    writer = cv2.VideoWriter("test1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        res = counter.feedFrame(image, True)
        
        if cv2.waitKey(1000//60) & 0xFF == 27:
            break

        writer.write(res)

    cap.release()

if __name__ == "__main__":
    main()
