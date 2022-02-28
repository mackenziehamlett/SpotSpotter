import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils
#from keras.preprocessing import image as pics

#------------------------------------------------------------------------------------------------------``
def find_contours(img): 
    contours, hierarchy = cv.findContours(image=img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # draw contours on the original image
    image_copy = img.copy()
    cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

    # see the results
    cv.imshow('Contour Approximation', image_copy)
    cv.waitKey(0)
    cv.imwrite('contours_none_image1.jpg', image_copy)
    cv.destroyAllWindows()

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

def is_contour_bad(c):
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)

    return not len(approx) == 4

# Detects potential cars based on how circular it is, returns boolean
def vehicle_detector_contour(contour_length, contour_area):
    return ((contour_length**2)/ contour_area) <= 20 # to roughly circular, like a car, adjust the value as needed

def main():
    image = cv.imread(r'C:\Users\ducke\parking_spots\download.jpg')
    cv.imshow("Original Image", image)

    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                          cv.THRESH_BINARY, 199, 5)
    t2 = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                          cv.THRESH_BINARY, 199, 5)
    cv.imshow('Adaptive Gaussian', thresh)
    cv.imshow('Mean', t2)

    find_contours(thresh)


    

if __name__ == "__main__":
    main()
