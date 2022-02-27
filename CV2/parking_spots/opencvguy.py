import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#------------------------------------------------------------------------------------------------------``
def find_contours(image, thresh): 

    contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # draw contours on the original image
    image_copy = image.copy()
    cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

    # see the results
    cv.imshow('Contour Approximation', image_copy)
    cv.waitKey(0)
    cv.imwrite('contours_none_image1.jpg', image_copy)
    cv.destroyAllWindows()


def hsv_split(image):
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    ret, thresh = cv.threshold, img_hsv, 179, 255, 255, cv.TH


def threshold_test():
    image = cv.imread(r'C:\Users\ducke\parking_spots\download.jpg')
    cv.imshow("Original Image", image)

    ret,thresh1 = cv.threshold(image,127,255,cv.THRESH_BINARY)
    ret,thresh2 = cv.threshold(image,127,255,cv.THRESH_BINARY_INV)
    ret,thresh3 = cv.threshold(image,127,255,cv.THRESH_TRUNC)
    ret,thresh4 = cv.threshold(image,127,255,cv.THRESH_TOZERO)
    ret,thresh5 = cv.threshold(image,127,255,cv.THRESH_TOZERO_INV)
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

def main():
    image = cv.imread(r'C:\Users\ducke\parking_spots\download.jpg')
    cv.imshow("Original Image", image)
    img = image.img_to_array(img, dtype='uint8')
    thresh = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
            
    find_contours(img, thresh)
    


    

if __name__ == "__main__":
    main()