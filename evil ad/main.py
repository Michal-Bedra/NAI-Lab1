from random import random

import cv2 as cv
import numpy as np
from ffpyplayer.player import MediaPlayer

"""
https://numpy.org/doc/stable/user/absolute_beginners.html
https://docs.opencv.org/4.x/db/dd1/tutorial_py_pip_install.html
https://matham.github.io/ffpyplayer/index.html
Filip Patuła s28615, Michał Bedra s28854
"""

HEIGHT = 480
WIDTH = 640
FRONTALFACE_DEFAULT_XML = './data/haarcascade_frontalface_default.xml'
HAARCASCADE_EYE_XML = './data/haarcascade_eye.xml'
ADVERTISEMENT_PATH = "panda.mp4"
CAMERA_PATH = "http://192.168.0.10:4747/video"

class AdvertisementSystem:
    """ Dobry system monitorowania użytkownika i odtwarzania reklamy"""
    def __init__(self):
        """ Ustawia podstawowe pola (wyświetlane obrazy, odtwarzacz, obiekt odtwarzania obrazu z kamery, klasyfikatory obrazu, inne parametry dla odtwarzania) w klasie potrzebne do działania systemu odtwarzania reklamy
            Parametry:
            self
            Zwraca:
            None
        """
        self.camera_delay = 40
        self.max_distraction_wait_time = 10
        self.max_x_pupil_eye_centre_difference = 4.2
        self.max_y_pupil_eye_centre_difference = 3.2
        self.base_volume = 0.1
        self.watch_img = self.create_watch_img()
        self.paused_img = self.create_paused_img()
        self.camera = self.setup_camera()
        self.advertisement_player = self.setup_advertisement_player()
        self.eye_cascade = cv.CascadeClassifier(HAARCASCADE_EYE_XML)
        self.face_cascade = cv.CascadeClassifier(FRONTALFACE_DEFAULT_XML)
        self.distraction_score = 0
        self.distraction_wait_timer = 0
        self.should_watch_me = True

    def volume_up(self):
        """ Zwiększa głośność odtwarzania reklamy na podstawie wartości pól
            Parametry:
            self
            Zwraca:
            None
        """
        volume = (self.base_volume + self.distraction_score / 1000)
        self.advertisement_player.set_volume(max(volume, 1.0))

    def create_paused_img(self):
        """ Tworzy obraz do wyświetlania w trakcie przerwania odtwarzania reklamy
            Parametry:
            self
            Zwraca:
            paused_img - tablica zawierająca wartości dla pikseli dla obrazu
        """
        paused_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv.putText(paused_img, "ARE YOU STILL WATCHING?", (int(WIDTH / 2 - 320), int(HEIGHT / 2)),
                   cv.FONT_HERSHEY_PLAIN, 3,
                   (0, 0, 255), 2)
        cv.putText(paused_img, "PRESS ANY KEY...", (int(WIDTH / 2 - 200), int(HEIGHT / 2 + 50)), cv.FONT_HERSHEY_PLAIN,
                   3,
                   (0, 0, 255), 3)
        return paused_img

    def create_watch_img(self):
        """ Pobiera i tworzy obraz do wyświetlania w trakcie wykrycia pogorszonego stanu oglądającego
            Parametry:
            self
            Zwraca:
            watch_text_img - tablica zawierająca wartości dla pikseli dla pobranego obrazu
        """
        watch_text_img = cv.imread("./uncle_sam.jpg")
        watch_text_img = cv.resize(watch_text_img, (WIDTH, HEIGHT))
        return watch_text_img

    def setup_camera(self):
        """ Otwiera strumień obrazu z kamery podłączonej do komputera
            Parametry:
            self
            Zwraca:
            camera - obiekt zarządzający strumieniem obrazów z kamery
        """
        camera = cv.VideoCapture(CAMERA_PATH)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        return camera

    def setup_advertisement_player(self):
        """ Otwiera strumień obrazu i dźwięku dla wczytanej reklamy możliwy do zarządznia
            Parametry:
            self
            Zwraca:
            advertisement_player - obiekt zarządzający strumieniem obrazów i dźwięku dla reklamy
        """
        advertisement_player = MediaPlayer(ADVERTISEMENT_PATH)
        advertisement_player.set_volume(self.base_volume)
        advertisement_player.set_size(WIDTH, HEIGHT)
        return advertisement_player

    def pause(self):
        """ Zatrzymuje reklamę i czeka na wciśnięcie przycisku przez użytkownika
            Parametry:
            self
            Zwraca:
            None
        """
        self.advertisement_player.set_pause(True)
        cv.imshow("Advertisement", self.paused_img)
        cv.waitKey(0)
        self.advertisement_player.set_pause(False)

    def pause_advertisement(self):
        """ Zatrzymuje reklamę na podstawie poziomu zadowolenia użytkownika
            Parametry:
            self
            Zwraca:
            None
        """
        if self.distraction_score > 2000:
            self.pause()
            self.distraction_score -= 500

    def adjust_volume(self):
        """ Dostosowuje głośność na podstawie poziomu zadowolenia użytkownika
            Parametry:
            self
            Zwraca:
            None
        """
        if self.distraction_score > 75:
            self.volume_up()
        else:
            self.advertisement_player.set_volume(self.base_volume)

    def show_advertisement_frame(self, advertisement_frame, val):
        """ Wyświetla na ekranie klatkę z reklamy (jeśli jest) odpowiednio dostosowaną na podstawie wartości zaangażowania użytkownika
            Parametry:
            self,
            advertisement_frame - klatka z reklamy
            val - czas wyświetlania klatki lub wartość eof
            Zwraca:
            None
        """
        if val != 'eof' and advertisement_frame is not None:
            img, t = advertisement_frame
            size = img.get_size()
            w = size[0]
            h = size[1]
            img_array = np.uint8(np.array(img.to_bytearray()[0]).reshape(h, w, 3))
            adv_image = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
            #Shows randomly watch img if user is not looking directly on the advertisement and his distraction level is medium
            if self.should_watch_me and self.distraction_score > 300 and random() < 0.2:
                adv_image = self.watch_img
            #Adds adjusted gaussian noise to the frame if user distraction level is high
            elif self.distraction_score > 1200:
                gaussian_noise = np.zeros((HEIGHT, WIDTH), dtype="uint8")
                cv.randn(gaussian_noise, 0.0, 15.0 + self.distraction_score / 50)
                gaussian_noise = cv.merge((gaussian_noise, gaussian_noise, gaussian_noise))
                adv_image = cv.add(adv_image, gaussian_noise)
            cv.imshow('Advertisement', adv_image)

    def detect_details_and_adjust_distraction_score(self, camera_frame):
        """ Sprawdza poziom zadowolenia użytkownika z oglądanej reklamy na podstawie obrazu z kamery, dostowuje wskaźniki zadowolenia i zaangażowania użytkownika
            Parametry:
            self,
            camera_frame - klatka z obrazu pobrana z kamery
            Zwraca:
            None
        """
        # Checks if frame from camera exists
        if not camera_frame is None:
            # Convert colors to grayscale
            gray_frame = cv.cvtColor(camera_frame, cv.COLOR_BGR2GRAY)
            gray_frame = cv.equalizeHist(gray_frame)
            # Search for faces
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            if len(faces):
                for (x, y, w, h) in faces:
                    face = gray_frame[y:y + h, x:x + w]
                    # Search for eyes
                    eyes = self.eye_cascade.detectMultiScale(face)
                    if len(eyes):
                        for (x2, y2, w2, h2) in eyes:
                            eye_x = x + x2
                            eye_y = y + y2
                            camera_frame = cv.rectangle(camera_frame, (eye_x, eye_y), (eye_x + w2, eye_y + h2),
                                                        (255, 0, 0), 2)
                            eye_roi = gray_frame[eye_y:eye_y + h2, eye_x:eye_x + w2]
                            # Search for pupil contours
                            _, eye_thresh = cv.threshold(eye_roi, 10, 255, cv.THRESH_BINARY_INV)
                            contours, _ = cv.findContours(eye_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                            if len(contours) != 0:
                                pupil = max(contours, key=cv.contourArea)
                                x3, y3, w3, h3 = cv.boundingRect(pupil)
                                camera_frame = cv.rectangle(camera_frame, (x3 + eye_x, y3 + eye_y),
                                                            (x3 + eye_x + w3, y3 + eye_y + h3), (0, 255, 0), 2)
                                pupil_centre_x = w3 / 2 + x3
                                pupil_centre_y = h3 / 2 + y3
                                eye_centre_x = w2 / 2
                                eye_centre_y = h2 / 2
                                # Calculate distance between eye center and pupil center
                                if (abs(eye_centre_x - pupil_centre_x) > self.max_x_pupil_eye_centre_difference) or (abs(eye_centre_y - pupil_centre_y) > self.max_y_pupil_eye_centre_difference):
                                    # Adjust engagement values
                                    self.distraction_wait_timer += 1
                                    if self.distraction_wait_timer > self.max_distraction_wait_time:
                                        self.distraction_score += 5
                                else:
                                    # Adjust engagement values, user is happy
                                    timer = self.distraction_wait_timer - 2
                                    self.distraction_wait_timer = timer if timer > 0 else 0
                                    score = self.distraction_score - 4
                                    self.distraction_score = score if score > 0 else 0
                                    self.should_watch_me = False
                            else:
                                # Adjust engagement values
                                self.distraction_wait_timer += 2
                                if self.distraction_wait_timer > self.max_distraction_wait_time:
                                    self.distraction_score += 10
                    else:
                        # Adjust engagement values
                        self.distraction_wait_timer += 2
                        if self.distraction_wait_timer > self.max_distraction_wait_time:
                            self.distraction_score += 10
            else:
                # Adjust engagement values
                self.distraction_wait_timer += 4
                if self.distraction_wait_timer > self.max_distraction_wait_time:
                    self.distraction_score += 15
            cv.imshow('Camera', camera_frame)
            cv.waitKey(self.camera_delay)

    def play_advertisement(self):
        """ Zarządza odtwarzeniem reklamy, obrazu z kamery i aktualizacją danych użytkownika
        Parametry:
        self
        Zwraca:
        None
        """
        while True:
            ret, camera_frame = self.camera.read()
            advertisement_frame, val = self.advertisement_player.get_frame()
            camera_frame = cv.rotate(camera_frame, cv.ROTATE_90_CLOCKWISE)
            self.should_watch_me = True
            if val == 'eof':
                break

            self.pause_advertisement()

            self.adjust_volume()

            self.detect_details_and_adjust_distraction_score(camera_frame)
            self.show_advertisement_frame(advertisement_frame, val)

            if val != 0.0 and cv.waitKey(int(val * 1000) - self.camera_delay) == ord('q'):
                break

        self.advertisement_player.close_player()
        self.camera.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    advertisement = AdvertisementSystem()
    advertisement.play_advertisement()