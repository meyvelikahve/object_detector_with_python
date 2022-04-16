import cv2  # open cv küüphanesi import edilmiştir


#gerçek zamanlı nesne algılama algoritmasını dışarıdan dosya olarak alıyoruz
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

#open cv nin DNN modülü sayesinde görüntü algılama algoritmasını parametre olarak veriyoruz
model = cv2.dnn_DetectionModel(frozen_model, config_file)  

#boş bir liste oluşturuyoruz
classLabels = []

#tespit ettiğimiz görüntüdeki nesne isimlerinin bulunduğu dosyayı tanımlıyoruz.
file_name = 'coco.names'
with open(file_name, 'rt') as f:                     # coco.names isimli dosyayı okumak içi açıyoruz
    classLabels = f.read().rstrip('\n').split('\n')  # içerideki verileri uygun hale getirip listeye atıyoruz


model.setInputSize(320, 320)                # giriş boyutunu sınırlandırıp ayarlıyoruz    
model.setInputScale(1.0 / 127.5)            # çerçeve değerleri için çarpan
model.setInputMean((127.5, 127.5, 127.5))   # kanallardan çıkarılan ortalama değerler
model.setInputSwapRB(True)                  # ilk ve son kanalı değiştiren method

cam = cv2.VideoCapture(1)                   # kameraya ulaşmamız için kullanılır

if not cam.isOpened():                      # kamera açıldıysa True değeri döndürür   
    cam = cv2.VideoCapture(0)               # default kameramıza geri dönüyoruz
if not cam.isOpened():
    raise IOError("Kamera açılamadı")


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame =cam.read()
    ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.55)
    print(ClassIndex)

    if(len(ClassIndex)!=0):
        for classInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            cv2.rectangle(frame, boxes, (255, 0, 0), 2)
            cv2.putText(frame, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                        color=(0, 255, 0), thickness=3)
    cv2.imshow('Test', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
