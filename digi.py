import cv2
import os
import numpy as np
# from keras import preprocessing
# from keras_vggface.vggface import VGGFace
import time
import sys
import matplotlib.pyplot as plt
import card


#ReadNetfromCaffe
#Config path
protoPath = os.path.join(os.getcwd(),'face_detector','deploy.prototxt.txt')
#Net path
modelPath = os.path.join(os.getcwd(),'face_detector','res10_300x300_ssd_iter_140000.caffemodel')
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#CNN
embedder = cv2.dnn.readNetFromTorch(os.path.join(os.getcwd(),'openface.nn4.small2.v1.t7'))
# vgg_features = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')
def get_embedding(img):
    blob = img_blob(img)
    detector.setInput(blob)
    detections = detector.forward()
    face_detected = False
    h,w ,c= img.shape
    if len(detections)>0:
        #Assume only one face
        i  = np.argmax(detections[0, 0, :, 2])
        
        box = detections[0,0,i,3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")
        
        id_face = img[startY:endY, startX:endX]
        id_face_blur = cv2.GaussianBlur(id_face, (5,5), 1)
        sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        id_face = cv2.filter2D(id_face_blur,-1,sharp_kernel)

        plt.imshow(cv2.cvtColor(id_face,cv2.COLOR_BGR2RGB))
        plt.show()

        id_face2 = face_blob(id_face)
        
        # embedded = vgg_features.predict(id_face2)
        embedder.setInput(id_face2)
        embedded = embedder.forward()
        face_detected = True
        
    # cv2.imshow(winname='face',mat=id_face)
    
    return face_detected, embedded

def selfie():
    #Number of pictures taken
    NUMBER = 5
    embedded_selfie = []
    frames = []
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == False:
        print("Can't connect to camera")
        return None
    
    #Is it capturing
    capturing = False
    #Time capturing started
    starttime = 0
    sec = 0
    while(cap.isOpened()):
        

        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
            if capturing and len(frames)<NUMBER:
                delta = time.time()-starttime
                if sec >= 3:
                    sec=0
                    starttime = starttime+3
                    print('Capturing')
                    frames.append(frame)
                    
                elif delta>sec:
                    print(str(3-sec)+'...')
                    sec += 1
                if len(frames) == 5:
                    break
            #Selfie on y key
            
            elif cv2.waitKey(1) == ord('y'):
                capturing = True
                starttime=time.time()
                print('Y')
            #Exit on q key
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    if len(frames) > 0:
        for img in frames:
            face_detected, embedded = get_embedding(img)
            if face_detected:
                embedded_selfie.append(embedded)
            
    return embedded_selfie

def id_card(path):
    id = cv2.imread(path)
    card_detect, id = card.extract_photo(id)
    if card_detect:
        face_detected, embedded_id = get_embedding(id)
        if not face_detected:
            print("No face detected")
            return None
        return embedded_id
    else:
        print("No card detected")
        return None

def img_blob(img):
    img = img.astype(np.float32)
    resized_img = cv2.resize(img,(300,300))
    # subtract image with RGB mean
    resized_img[:,:,0] -= 104.0
    resized_img[:,:,1] -= 177.0
    resized_img[:,:,2] -= 123.0

    resized_img = np.transpose(resized_img, [2, 0, 1])
    resized_img = np.array([resized_img])
    
    return resized_img

def face_blob(img):
    img = img*(1/255.0)
    swapRB = img[:,:,::-1]
    # resized_img = cv2.resize(swapRB,(224,224))
    # resized_img[:,:,0] += 93.5940 
    # resized_img[:,:,1] += 103.8827
    # resized_img[:,:,2] += 129.1863
    # resized_img = np.asarray((resized_img,))
    resized_img = cv2.resize(swapRB, (96,96))
    resized_img = np.transpose(resized_img, [2, 0, 1])
    resized_img = np.array([resized_img])
    return resized_img

def compare_face(embedded_id,embedded_selfies,threshold=0.85):
    if len(embedded_id) == 0 or len(embedded_selfies) == 0:
        euclidean_distance = np.empty((0))
    matches = 0 
    for embedded_img in embedded_selfies: #(1,128)
        euclidean_distance2 = np.linalg.norm(embedded_img[0]-embedded_id[0])
        euclidean_distance1 = np.sqrt(np.mean((embedded_img[0]-embedded_id[0])**2))
        print(euclidean_distance2)
        if euclidean_distance2 <= threshold:
            matches +=1
    return matches > len(embedded_selfies)/2
    

def main():
    embedded_id = id_card(sys.argv[1])
    if not embedded_id is None:
        embedded_selfies = selfie()
        if len(embedded_selfies) == 0:
            print("No face detected in selfies")
            return
        if embedded_selfies != []:
            matched = compare_face(embedded_id,embedded_selfies)
            if matched:
                print("Pass")
            else:
                print("Try Again")
        else:
            print('no face detected')
    

if __name__ == '__main__':
    main()
