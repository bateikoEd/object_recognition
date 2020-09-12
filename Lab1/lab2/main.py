import numpy as np
import cv2


def create_video(video_file_name='output.avi'):
    """Create,flip and save the video"""

    video_file_name = f'files/{video_file_name}'
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(video_file_name, fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # flip video
            frame = cv2.flip(frame, 0)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def edit_video(name_file_video='output.avi', saved_name_file_video='output_edited.avi'):
    """Edit the loaded video file as gray color,
    and create rectangle and line on the video"""

    name_file_video = f'files/{name_file_video}'
    saved_name_file_video = f'files/{saved_name_file_video}'

    cap = cv2.VideoCapture(name_file_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(saved_name_file_video, fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            cv2.rectangle(gray, (200, 200), (300, 300), (0, 100, 255), 20)
            cv2.line(gray, (60, 20), (350, 200), (0, 100, 255), 5)

            cv2.imshow('Edited video', gray)
            out.write(gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    create_video()
    edit_video('output.avi')