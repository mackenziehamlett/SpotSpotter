import cv2 as cv

def grayscale_and_find_contours(image): 
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
    cv.imshow('Binary Image', thresh)
    #cv.waitKey(0)
    cv.imwrite('image_thres1.jpg', thresh)
    #cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # draw contours on the original image
    image_copy = img_gray.copy()
    cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

    # see the results
    cv.imshow('Contour Approximation', image_copy)
    cv.waitKey(0)
    cv.imwrite('contours_none_image1.jpg', image_copy)
    cv.destroyAllWindows()

def main():
    image = cv.imread(r'C:\Users\ducke\parking_spots\download.jpg')
    cv.imshow("Original Image", image)
    #grayscale_and_find_contours(image)
    

    

if __name__ == "__main__":
    main()
