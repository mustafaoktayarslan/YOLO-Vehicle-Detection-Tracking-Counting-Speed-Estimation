import math
from ultralytics import YOLO
import cv2
import cvzone
from sort import *
from tkinter import messagebox, simpledialog
import keyboard

# "https://izum.izmir.bel.tr/apps/cameras/detail.html"
mobese_url = 'https://izum-cams.izmir.bel.tr/mjpeg/a4808057-b7a5-4b69-9d52-fba73f597703'
mobese_url2 = 'https://izum-cams.izmir.bel.tr/mjpeg/1b4a6338-b0b3-435b-a26a-25222cf0be08'
mobese_url3 = 'https://izum-cams.izmir.bel.tr/mjpeg/a5f1b361-48fa-42f5-ac43-86409bac9635'
mobese_url4 = 'https://izum-cams.izmir.bel.tr/mjpeg/df463759-3be8-4630-8de1-de1f7185a38c'
mobese_url5 = 'https://izum-cams.izmir.bel.tr/mjpeg/3a581c91-3506-484c-8444-4ea06b14410f'
mobese_url6 = 'https://izum-cams.izmir.bel.tr/mjpeg/8aa35cea-328f-4e93-b51f-7559cda13a23'
mobese_url7 = 'https://izum-cams.izmir.bel.tr/mjpeg/282ac058-5097-4b44-bda8-47167c9786db'
mobese_url8 = 'https://izum-cams.izmir.bel.tr/mjpeg/50b6305c-10ae-4025-9ba2-3e734738ae9a'

classNames = [
    "insan", "bicycle", "araba", "motorsiklet", "airplane", "otobus", "train", "kamyonet",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
# 822,550  1260,542
x1, y1, x2, y2 = (0, 0, 0, 0)
totalCount = set()
clickCount = 0
ytemp = 0
lines = []
limitses = []
limitss = []
object_speeds = {}

prev_frame_time = 0
new_frame_time = 0
sayim = True
hiz = True


class TrafficDensity:
    def __init__(self):
        self.total_count = 0
        self.start_time = time.time()
        self.density = 0.0
    def update_density(self, current_count):
        elapsed_time = time.time() - self.start_time
        self.total_count = current_count
        # Güncellenmiş yoğunluğu hesapla (örneğin, araç sayısı / geçen süre)
        if elapsed_time > 0:
            self.density = self.total_count / elapsed_time
        else:
            self.density = 0.0
    def get_density(self):
        return self.density


class CustomLine:
    def __init__(self, start_point, end_point, color, thickness):
        self.start_point = start_point
        self.end_point = end_point
        self.color = color
        self.thickness = thickness

    def draw_on_image(self, image):
        cv2.line(image, self.start_point, self.end_point, self.color, self.thickness)


class Rectangle:
    def __init__(self):
        self.object_inside = False

    def check_collision(self, line1, line2, object_x, object_y):

        min_x = min(line1[0], line1[2], line2[0], line2[2])
        max_x = max(line1[0], line1[2], line2[0], line2[2])
        min_y = min(line1[1], line1[3], line2[1], line2[3])
        max_y = max(line1[1], line1[3], line2[1], line2[3])

        # Noktanın bu dikdörtgen içinde olup olmadığını kontrol et
        if min_x <= cx <= max_x and min_y <= cy <= max_y:
            self.object_inside = True
            return True
        else:
            self.object_inside = False
            return False
def mouse_callback(event, x, y, flags, param):
    global clickCount, limitss
    if event == cv2.EVENT_LBUTTONDOWN:
        clickCount = (clickCount + 1) % 2
        global ytemp
        if (clickCount == 1):

            messagebox.showinfo("Koordinat 1", f"Mouse tıklandı - Koordinatlar: ({x}, {y})")
            limitss = [x, y, 0, 0]
            ytemp = y
        elif (clickCount == 0):
            messagebox.showinfo("Koordinat 2", f"Mouse tıklandı - Koordinatlar: ({x}, {y})")
            limitss[2], limitss[3] = x, ytemp
            line = CustomLine((limitss[0], limitss[1]),
                              (limitss[2], limitss[3]),
                              (0, 0, 255), 4)
            print('click', limitss)
            lines.append(line)
            limitses.append(limitss.copy())

# camera = cv2.VideoCapture('https://youtu.be/Xjj_hXPwGqE')
# camera.set(3, 1280)
# camera.set(4, 720)
# camera = cv2.VideoCapture(0)

camera = cv2.VideoCapture("../videolar/otoyol1.mp4")
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

#mobese_url = 'https://izum-cams.izmir.bel.tr/mjpeg/a4808057-b7a5-4b69-9d52-fba73f597703'
#camera = cv2.VideoCapture(mobese_url)

new_width, new_height = 1280, 720
camera.set(3, new_width)
camera.set(4, new_height)

fpsgelen = camera.get(cv2.CAP_PROP_FPS)

mask = cv2.imread("maskesol.png")
# mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
a, b, c, d = 400, 50, 700, 50
panel = cv2.imread("./panel.png", cv2.IMREAD_UNCHANGED)
countcikis = 0
countgiris = 0
traffic_density = TrafficDensity()
# Not: Bu değeri ölçmek size düşer, aşağıdaki örnekte 12 metre kullanılmıştır
real_world_distance = 12.0  # Ölçülen gerçek dünya mesafesi (metre cinsinden)

model = YOLO('yolov8n.pt')
while True:
    success, img = camera.read()
    if not success or img is None:
        print("Hata: Görüntü okunamadı veya boş.")
        break

    # FPS'i hesaplıyoruz.
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    img = cv2.resize(img, (new_width, new_height))
    img_width = img.shape[1]
    img_height = img.shape[0]
    # imgRegion = cv2.bi    twise_and(img, mask)

    img = cvzone.overlayPNG(img, panel, (0, 0))
    results = model(img, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes

        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            classIndex = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]
            #  bx1,by1,w,h=box.xywh[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # print(x1, y2, x2, y2)
            #  cv2.rectangle(img,(x1,y1),(x2,y2),(200,0,0),3)
            currentClass = classNames[classIndex]
            # print("conf:", conf)
            #  print("conf:", currentClass)
            #  print("\n")
            conftext = int(conf * 100)
            denemey = int(y1 + w + 3)
            # print("CLASS INDEKS", classIndex)

            if (currentClass == "araba" or currentClass == "otobus"
                or currentClass == "kamyonet" or currentClass == "motorsiklet" and conf > 0.4):
                cvzone.putTextRect(img, f'{currentClass} | %{conftext} ',
                                   (min(img_width, int(x1)), min(img_height, int(y1 - 5))),
                     scale=1,thickness=1, offset=2,colorR=(0, 0, 255))


                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    previous_positions = {}
    resultsTracker = tracker.update(detections)

    for index, line in enumerate(lines, start=1):
        line.draw_on_image(img)
        px1, py1 = line.start_point
        px2, py2 = line.end_point
        if index==1 and sayim:
            cvzone.putTextRect(img, f'{index} IN', (px2, py2), scale=2,
                               thickness=2, offset=2, colorR=(0, 255, 0))
        elif index==2 and sayim:
            cvzone.putTextRect(img, f'{index} OUT', (px2, py2), scale=2,
                               thickness=2, offset=2, colorR=(0, 255, 0))
        else:
            cvzone.putTextRect(img, f'{index} ', (px2, py2), scale=2,
                           thickness=2, offset=2, colorR=(0, 255, 0))

    for result in resultsTracker:
        x1, y1, x2, y2, id = result

        #    print("result: ",result)
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = int(x1 + w // 2), int(y1 + h // 2)

        if id in object_speeds and object_speeds[id]["speed"] > 0:
            cvzone.putTextRect(img, f'#{id}, Hiz: {object_speeds[id]["speed"]} km/h',
                               (min(img_width, int(x1)), min(img_height, int(y2 + 15))), scale=2,
                               thickness=2, offset=2,
                               colorR=(255, 0, 0))
        else:
            cvzone.putTextRect(img, f'#{id}',
                               (min(img_width, int(x1)), min(img_height, int(y2 + 15))), scale=1,
                               thickness=1, offset=2,
                               colorR=(255, 0, 0))

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=3, colorR=(255, 0, 255), rt=1)

        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        for limits in limitses:
            # print(limits)
            if not limits == [0, 0, 0, 0]:
                if len(limitses) >= 2 and hiz:
                    rx1, ry1, rx2, ry2 = limitses[0][0], limitses[0][1], limitses[0][2], limitses[0][3]
                    rectangle = Rectangle()
                    rectanglein = rectangle.check_collision(limitses[0], limitses[1], cx, cy)

                    if id in object_speeds and rectanglein:
                        if object_speeds[id]["has_crossed_line"] == True:
                            ref_point1 = (
                                limitses[0][0], limitses[0][1])  # İlk referans noktasının pixel koordinatları
                            ref_point2 = (
                                limitses[1][0], limitses[1][1])  # İkinci referans noktasının pixel koordinatları
                            # İki referans noktasının pixel cinsinden mesafesini ölçün
                            ref_distance_pixels = np.linalg.norm(np.array(ref_point2) - np.array(ref_point1))
                            # İki referans noktasının gerçek dünya mesafesini ölçün
                            # Ölçek faktörünü hesaplayın: Pixel cinsinden mesafe ile gerçek dünya mesafesi arasındaki oran
                            pixel_to_real_world_factor = real_world_distance / ref_distance_pixels

                            prev_x, prev_y, prev_time = object_speeds[id]["position"]
                            has_crossed_line = object_speeds[id]["has_crossed_line"]
                            distance = np.linalg.norm(np.array([cx, cy]) - np.array([prev_x, prev_y]))
                            distance_real_world = distance * pixel_to_real_world_factor
                            time_diff = time.time() - prev_time
                            speed = distance_real_world / time_diff
                            # km/h cinsine çevirmek
                            speed_kmh = speed * 3.6
                            # kameradan gelen fps ile işleme fpsini oranlarız
                            speed_kmh = speed_kmh * fpsgelen / fps
                            if not math.isnan(speed_kmh):
                                object_speeds[id]["speed"] = int(speed_kmh)
                            else:
                                # `speed_kmh` NaN ise, burada başka bir şey yapabilirsiniz (örneğin varsayılan bir değer atama)
                                object_speeds[id][
                                    "speed"] = 0  # varsayilan_deger'i uygun bir değerle değiştirin
                            print(f"***FPS:{fps}, !!!gelenfps= {fpsgelen}")



                    else:
                        object_speeds[id] = {"position": (cx, cy, time.time()), "speed": 0,
                                             "has_crossed_line": True}

                if limits[0] <= cx <= limits[2] and limits[1] - 20 < cy < limits[3] + 15:

                    if id not in totalCount:
                        if len(limitses) >= 2 and sayim:
                            if limitses[0] == limits:
                                countgiris = countgiris + 1
                                print("GİRİŞ", countgiris)

                            if limitses[1] == limits:
                                countcikis = countcikis + 1
                                print("ÇIKIŞ", countcikis)
                        totalCount.add(id)
                        if id in object_speeds and "has_crossed_line" in object_speeds[id]:
                            object_speeds[id]["has_crossed_line"] = True

                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 8)
                        cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)


                elif limits[0] < cx < limits[2] and limits[3] - 15 < cy < limits[1] + 15:

                    if id not in totalCount:
                        totalCount.add(id)
                        if len(limitses) >= 2 and sayim:
                            if limitses[0] == limits:
                                countgiris = countgiris + 1
                                print("GİRİŞ", countgiris)

                            if limitses[1] == limits:
                                countcikis = countcikis + 1
                                print("ÇIKIŞ", countcikis)
                        if id in object_speeds and "has_crossed_line" in object_speeds[id]:
                            object_speeds[id]["has_crossed_line"] = True
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 8)
                        cv2.circle(img, (cx, cy), 18, (255, 0, 255), cv2.FILLED)


                elif limits[2] < cx < limits[0] and limits[1] - 15 < cy < limits[3] + 15:

                    if id not in totalCount:
                        if len(limitses) >= 2 and sayim:
                            if limitses[0] == limits:
                                countgiris = countgiris + 1
                                print("GİRİŞ", countgiris)

                            if limitses[1] == limits:
                                countcikis = countcikis + 1
                                print("ÇIKIŞ", countcikis)
                        totalCount.add(id)
                        if id in object_speeds and "has_crossed_line" in object_speeds[id]:
                            object_speeds[id]["has_crossed_line"] = True
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 8)
                        cv2.circle(img, (cx, cy), 18, (255, 0, 255), cv2.FILLED)

                        # totalCount şeklinde tuttuğumuzda arayı sürekli taradığı için denetimsiz artıyor.
                    # totalCount+=1
                elif limits[2] < cx < limits[0] and limits[3] - 15 < cy < limits[1] + 15:
                    if id not in totalCount:
                        totalCount.add(id)
                        if len(limitses) >= 2 and sayim:
                            if limitses[0] == limits:
                                countgiris = countgiris + 1
                                print("GİRİŞ", countgiris)

                            if limitses[1] == limits:
                                countcikis = countcikis + 1
                                print("ÇIKIŞ", countcikis)
                        if id in object_speeds and "has_crossed_line" in object_speeds[id]:
                            object_speeds[id]["has_crossed_line"] = True
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 8)
                        cv2.circle(img, (cx, cy), 18, (255, 0, 255), cv2.FILLED)

    text = f'{len(totalCount)}'
    text_position = (94, 47)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 3
    font_thickness = 5
    font_color = (0, 0, 0)  # Beyaz renk (B, G, R) formatında
    rect_color = (255, 255, 255)
    # OpenCV'nin rectangle fonksiyonunu kullanarak arka plan dikdörtgenini ekleyelim
    cvzone.putTextRect(img, text, text_position, font_scale, font_thickness, font_color, rect_color, offset=17)

    textt = f'IN: {countgiris} OUT: {countcikis}'
    text_position = (94, 47)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    font_thickness = 2
    font_color = (0, 0, 0)  # Beyaz renk (B, G, R) formatında
    rect_color = (255, 255, 255)
    # OpenCV'nin rectangle fonksiyonunu kullanarak arka plan dikdörtgenini ekleyelim
    cvzone.putTextRect(img, textt, (new_width - 150, 25), font_scale, font_thickness, font_color, rect_color, offset=17)

    traffic_density.update_density(len(totalCount))
    current_density = traffic_density.get_density()
    traffic_state = "AZ"
    if current_density <= 0.3:
        traffic_state = "AZ"
    elif current_density > 0.3 and current_density <= 0.6:
        traffic_state = "ORTA"
    elif current_density > 0.6 and current_density <= 1:
        traffic_state = "COK"
    elif current_density > 1:
        traffic_state = "ASIRI"

    cvzone.putTextRect(img, f'Trafik Yogunlugu: {current_density:.2f} | {traffic_state}',
                (500, 25), font_scale,font_thickness, font_color, rect_color, offset=17)

    # cv2.putText(img, text,text_position, font, font_scale, font_color, font_thickness)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion",imgRegion)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('a'):
        limits = [0, 0, 0, 0]

    if keyboard.is_pressed('ctrl+d'):
        index_to_delete = simpledialog.askinteger("Silme Islemi", "Hangi cizgiyi silmek istersiniz:")

        # İndeks sınırları kontrol ediliyor.
        if index_to_delete is not None and 1 <= index_to_delete <= len(lines):
            index_to_delete -= 1
            lines.pop(index_to_delete)
            limitses.pop(index_to_delete)
            #  del lines[(index_to_delete-1)]
            # del limitses[(index_to_delete-1)]
            print(f"{lines}. indeks silindi.")
        else:
            print("Geçersiz indeks.")
    if keyboard.is_pressed('ctrl+r'):
        acitive_pasife = simpledialog.askinteger("Aktif/Pasif Islemi",
                                                 "Hangi özelliği aktif/pasif etmek istersiniz \n1-HIZ \n2- GİRİŞ ÇIKIŞ")
        # İndeks sınırları kontrol ediliyor.
        if acitive_pasife == 1:
            if hiz == True:
                hiz = False
            else:
                hiz = True
        elif acitive_pasife == 2:
            if sayim == True:
                sayim = False
            else:
                sayim = True
        else:
            print("Geçersiz indeks.")
        print(f'SAYIM:{sayim} HIZ: {hiz}')
    if keyboard.is_pressed('ctrl+m'):
        distanceIn = simpledialog.askinteger("Mesafe Islemi", "Mesafeyi Giriniz.")
        if distanceIn > 0:
            real_world_distance = distanceIn
        else:
            print("Mesafe ayarlanamadı")

        # İndeks sınırları kontrol ediliyor.
camera.release()
cv2.destroyAllWindows()
