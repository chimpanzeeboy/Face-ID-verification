import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

#Extract photo from card by position
def extract_photo(card):
    card_width, card_height = (645, 405)
    card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    card_blur = cv2.GaussianBlur(card_gray, (5,5), 1)
    #Use Canny Edge detection and morphology
    card_edge = cv2.Canny(card_blur, 200,250)
    close_kernel = np.ones((5,5))
    card_edge = cv2.morphologyEx(card_edge,cv2.MORPH_CLOSE, close_kernel)
    # plt.imshow(card_edge,cmap='gray')
    # plt.show()

    #Find_Countour
    biggest_contour = find_biggest_countours(card_edge,card)
    big_con_img = card.copy()
    cv2.drawContours(biggest_contour,biggest_contour,-1,(0,255,0),5)
    plt.imshow(big_con_img,cmap='gray')
    plt.show()
    photo = np.zeros((540,860,3))
    #Change perspective
    if biggest_contour.size != 0:
        edge_points = rearrange(biggest_contour)
        result_perspective = np.float32([[0,0], [card_width,0],[0,card_height],[card_width,card_height]])
        from_perspective = np.float32(edge_points)
        matrix = cv2.getPerspectiveTransform(from_perspective, result_perspective)
        extracted_card = cv2.warpPerspective(card,matrix,(card_width,card_height))
        extracted_card = cv2.resize(extracted_card,(int(860*1.5),int(540*1.5)))
        sharp_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        card_sharp = cv2.filter2D(extracted_card,-1,sharp_kernel)
        photo = card_sharp[370:700,950:]
    return photo

def rearrange(points):
    points = points.reshape((4,2))
    new_points = np.zeros((4,1,2),dtype=np.int32)
    #Top left will have lowest sum, Bottom right will have highest sum, Top right will have lowest diff, Lower left will have highest diff
    add = points.sum(1)
    diff = np.diff(points,axis=1)
    #Add to new points
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]


    return new_points  
    


def find_biggest_countours(card_edge,card):
    contours, hierarchy = cv2.findContours(card_edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    card_contours = card.copy()
    cv2.drawContours(card_contours,contours,-1,(0,255,0),10)
    cv2.imshow('card_con',card_contours)
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 3000:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.1*peri,True)
            print(approx)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest
def main():
    #Read from image
    card = cv2.imread(sys.argv[1])
    photo = extract_photo(card)
    print(photo.shape)
    plt.imshow(cv2.cvtColor(photo,cv2.COLOR_BGR2RGB))
    print('after')
    plt.show()


if __name__ == '__main__':
    main()    
    

