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

    return image_copy
    # see the results
    """cv.imshow('Contour Approximation', image_copy)
    cv.waitKey(0)
    cv.imwrite('contours_none_image1.jpg', image_copy)
    cv.destroyAllWindows()"""

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


def canny_edges(img):
    """Turn image to grey and blur, create edges from image"""
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (3, 3), 0)
    #thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 199, 5)
    
    edges = cv.Canny(blur, threshold1=100, threshold2=200) # Canny edge
    
    start_point = (112, 85)
    end_point = (163, 173)

    inv_edge = np.invert(edges)
    drawing = cv.rectangle(inv_edge, end_point, start_point, (0, 0, 255), thickness=3, lineType=cv.LINE_8)

    cv.imshow("Testing Baby", drawing)
    cv.waitKey(0)



    return None


def hough_line():
    img = cv.imread('download.jpg', cv.IMREAD_COLOR)

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    
    edges = cv.Canny(grey, threshold1=50, threshold2=200)

    lines = cv.HoughLinesP(edges, 1, np.pi/180, 50,  minLineLength=10, maxLineGap=250)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv.imshow("Lines", img)
    cv.waitKey(0)

def contours_and_edges():
    """
    image2 = cv.imread('input/custom_colors.jpg')
    contours4, hierarchy4 = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    image_copy5 = image2.copy()
    cv.drawContours(image_copy5, contours4, -1, (0, 255, 0), 2, cv.LINE_AA)
    # see the results
    cv.imshow('EXTERNAL', image_copy5)
    print(f"EXTERNAL: {hierarchy4}")
    cv.waitKey(0)
    cv.imwrite('contours_retr_external.jpg', image_copy5)
    cv.destroyAllWindows()
    image2 = cv2.imread('input/custom_colors.jpg')
    """

    image = cv.imread('download.jpg')
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(grey, 130, 255, 1)

    cnts = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        approx = cv.approxPolyDP(c, 0.009 * cv.arcLength(c, True), True)
        cv.drawContours(image, [c], 0, (0,255,0),3)


    # Used to flatted the array containing
    # the co-ordinates of the vertices.
    img2 = cv.imread('download.jpg', cv.IMREAD_COLOR)
    font = cv.FONT_HERSHEY_COMPLEX
    n = approx.ravel() 
    i = 0
  
    for j in n :
        if(i % 2 == 0):
            x = n[i]
            y = n[i + 1]
  
            # String containing the co-ordinates.
            string = str(x) + " " + str(y) 
  
            if(i == 0):
                # text on topmost co-ordinate.
                cv.putText(img2, "Arrow tip", (x, y),font, 0.5, (255, 0, 0)) 
            else:
                # text on remaining co-ordinates.
                cv.putText(img2, string, (x, y), font, 0.5, (0, 255, 0)) 
        i = i + 1

    cv.imshow("result", image)
    cv.waitKey(0)

def main():
    image = cv.imread(r'C:\Users\ducke\parking_spots\download.jpg')
    
    #canny_edges(image)
    contours_and_edges()
    #hough_line()
    cv.waitKey(0)
    #hough_line()


    #cv.imshow("edges", i2)
    #cv.waitKey(0)


    #print(canny_edges.__doc__)





    

if __name__ == "__main__":
    main()
