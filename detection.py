
import cv2
import numpy as np
from frst import frst

from preprocess import denoising
from utils import show_img
import os

def find_circles(img):
    # how well does it track the counters?
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,51)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    # circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
    #                             param1=50,param2=30,minRadius=0,maxRadius=10)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20, param1=30,param2=15,minRadius=0,maxRadius=50)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    show_img(cimg)

def draw_circles(img, circles):
    for i in circles:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(255,0,255),3)
    return img

def drawLinesP(img, lines):

    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
    return img

def run_kmeans(data):
    if data.dtype != np.float32:
        data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(data,2,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    return label

def detectLinesP(img_in):
    temp_img = np.copy(img_in)
    gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    minLineLength = 100
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
    return lines, temp_img

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """

    # from tempfile import TemporaryFile
    # outfile = TemporaryFile()

    temp_img = np.copy(img_in)
    cimg = np.copy(img_in[:,:,2])
    # np.save(outfile, cimg)

    np.putmask(cimg, np.logical_or(cimg>245, cimg<200), 0)

    show_img(cimg)

    cimg = cv2.medianBlur(cimg,11)
    # cv2.blur(cimg,(10,10))
    # gray = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cimg,50,150,apertureSize = 3)
    show_img(edges)

    minLineLength = 10
    maxLineGap = 7
    lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap)

    mid_x, mid_y = 0, 0
    stop_sign = None
    if lines is not None and lines.shape[1] > 8:
        #print 'lines', lines
        lines = lines[0]
        labels = run_kmeans(lines)

        zeros = lines[labels.ravel()==0]
        ones = lines[labels.ravel()==1]

        if len(zeros)==8:
            #print 'stop sign detected'
            stop_sign = zeros
        elif len(ones)==8:
            #print 'stop sign detected'
            stop_sign = ones

        temp_img = drawLinesP(temp_img, np.asarray([lines]))
    # lines, temp_img = detectLinesP(img_in)
        show_img(temp_img)
    # temp_img = np.copy(img_in[:,:,2])
    # cimg = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
    # cimg = cv2.medianBlur(cimg,3)
    # cimg = cv2.GaussianBlur(temp_img,(5,5),0)
    # show_img(cimg)
    # #print cimg
    cimg = np.copy(img_in[:,:,2])
    # np.save(outfile, cimg)

    np.putmask(cimg, np.logical_or(cimg>245, cimg<200), 0)

    cimg = cv2.medianBlur(cimg,51)
    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20, param1=30,param2=15,minRadius=0,maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        #print 'circles:', circles

        # circles = circles[0][-1]
        # mid_x = circles[1]
        # mid_y = circles[0]
        # pixelSize = 12

        # right_x, left_x = mid_x - pixelSize, mid_x + pixelSize
        # top_y, bottom_y = mid_y - pixelSize, mid_y + pixelSize
        # #print right_x, left_x, top_y, bottom_y
        # red = img_in[right_x:left_x, top_y:bottom_y, 2]
        # green = img_in[right_x:left_x, top_y:bottom_y, 1]
        # blue = img_in[right_x:left_x, top_y:bottom_y, 0]
        #
        #
        # #print red
        #
        # #print cimg[right_x:left_x, top_y:bottom_y]
        # show_img(img_in[right_x:left_x, top_y:bottom_y,:])
        # show_img(cimg[right_x:left_x, top_y:bottom_y])
        # #print np.asarray([lines[0][0],lines[0][2]])
        temp_img = draw_circles(temp_img, circles[0])
    else:
        print('no circle')


    # show_img(temp_img)

    # temp_img = np.copy(img_in)
    # temp_img = drawLinesP(temp_img, zeros)
    # show_img(temp_img)
    #
    #
    # temp_img = np.copy(img_in)
    # temp_img = drawLinesP(temp_img, ones)
    # show_img(temp_img)
    # temp_img = draw_circles(img_in, circles[0])

    # temp_img = detectLines(img_in)
    show_img(temp_img)
    if circles is not None:
        mid_x, mid_y = circles[0][0][:2]

    return mid_x, mid_y

def find_radial_symmetry(img):
    # find regions of interest
    firstImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    firstImg = frst(firstImg, 200, 1, 0.5, 0.1, 'BRIGHT')
    # show_img(firstImg)

    firstImg = cv2.normalize(firstImg, None, 0.0, 1.0, cv2.NORM_MINMAX)
    # frstImage.convertTo(frstImage, CV_8U, 255.0);

    firstImg = cv2.convertScaleAbs(firstImg, None, 255.0, 0)
    show_img(firstImg)


    ret, thresh = cv2.threshold(firstImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # show_img(ret)
    # cv::morphologyEx(inputImage, inputImage, operation, element, cv::Point(-1, -1), iterations);
    # bwMorph(frstImage, markers, cv::MORPH_CLOSE, cv::MORPH_ELLIPSE, 5);

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    markers =  cv2.morphologyEx(firstImg,cv2.MORPH_CLOSE, kernel, iterations=5)

    print(len(markers))

    im2, contours, hierarchy = cv2.findContours(markers,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    #
    # img = cv2.drawContours(img, contours, -1, (0,0,255), 8)
    # show_img(img)
    moments = []
    for cnt in contours:
        moment = cv2.moments(cnt)
        moments.append(moment)
        print(moment)


    #get the mass centers
    mass_centers = []
    for moment in moments:
        x, y = moment.get('m10') / moment.get('m00'), moment.get('m01') / moment.get('m00')
        mass_centers.append((int(x),int(y)))

    for center in mass_centers:
        print(center)
        cv2.circle(img, center, 20, (0,255,0), 2)
    show_img(img)
    return None

def HSV_IMAGE(img):
    pass

def select_ROI(img):
    r = cv2.selectROI(img)
    print(r)
    # Crop image
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    show_img(imCrop)
    print(imCrop.shape)
    return imCrop


class HAARDetection(object):

    def __init__(self, path=None):
        if path:
            if os.path.exists(path):
                print('reading detector from ', path)
                self.detector = cv2.CascadeClassifier(path)

        else:
            print('No detector found, recalculating')


    def find_signs(self, in_img):

        img = np.copy(in_img)
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img =denoising(img)
        whirldata = self.detector.detectMultiScale(gray_img, 20, 20)

        return whirldata
         # add this
        # print(whirldata)
        candidate_signs = []
        for (x,y,w,h) in whirldata:
             just_box = in_img[x:x+w,y:y+h,:]
             box_h, box_w, _ = just_box.shape
             if box_h == 0 or box_w == 0:
                 continue
             # print('box shape', just_box.shape)
             # print(x,x+w, y, y+h)
             show_img(just_box)
             candidate_signs.append(just_box)
             # candidate_signs.append(just_box)
             # candidate_signs.append(just_box)
             # candidate_signs.append(just_box)
             # candidate_signs.append(just_box)
             font = cv2.FONT_HERSHEY_SIMPLEX
             cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
             # cv2.putText(img, 'Sign',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
        show_img(img)
        # print(type(candidate_signs))
        print(len(candidate_signs))
        #
        # for region in candidate_signs:
        #     # print(region)
        #     print('region shape in', region.shape)

        return candidate_signs




