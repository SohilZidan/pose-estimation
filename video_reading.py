import cv2
import time

if __name__ == '__main__':
    # 1 second
    t_end = time.time() + 3 * 1
    # frames counter
    i = 0
    video = cv2.VideoCapture("./video_reading.mp4")
    while (t_curr := time.time()) < t_end and video.isOpened():
        # process the frame
        ret, frame = video.read()
        if ret == False:
            break
        #cv2.imwrite('kang' + str(i) + '.jpg', frame)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # j = 0
        # while j < 1000000:
        #     2^2^2^2^2
        #     j+=1
        i+=1;
        #time.sleep(1)
        # print(t_curr)
        # print(t_end)

    print("Frames per second for the video: {0} fps".format(video.get(cv2.CAP_PROP_FPS)))
    print("inference performance for the model: {0} fps".format(i))
    video.release()
    cv2.destroyAllWindows()