import cv2
import numpy as np
from collections import defaultdict

from detection import HAARDetection
from svm_utils import saved_svm_or_fit, classify, saved_svm_or_fit_bin, get_saved_svms
from utils import show_img, load_img_from_annotation, imgs_to_video, getFiltersHistograms
from path_collection import  *
from svm_utils import  getHogDescriptor




def get_images_of_type(path, signType=None, only_get=10):

    num = 0
    images_of_interest = []
    print(path)
    for labelPath in os.listdir(path):
        imagePath = os.path.join(path, labelPath)
        print('has', imagePath)
        if '.png' in imagePath:
            if signType:
                if signType in imagePath:
                    # print(imagePath)
                    # img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
                    img = cv2.imread(imagePath)
                    # if num < 5:
                    #     show_img(img, signType)

                    img = cv2.resize(img, (640,480))
                    num += 1
                    images_of_interest.append(img)
                    if num > only_get:
                        break
            else:
                # print(imagePath)
                # img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
                img = cv2.imread(imagePath)
                # if num < 5:
                #     show_img(img, signType)

                img = cv2.resize(img, (640,480))
                num += 1
                images_of_interest.append(img)
                if only_get and num > only_get:
                    break
    print(len(images_of_interest))
    return np.array(images_of_interest)



def faceDetection(image, classifier, hog, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        height, width = image.shape[:2]
        stride = 32
        x=0
        y=0
        img = np.copy(image)
        # gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        test_images = []
        x_rects=[]
        y_rects=[]
        while y+stride < height:
            test_img = img[y:y+stride, x:x+stride]
            test_img = hog.compute(test_img).flatten()
            test_images.append(test_img)
            # print(prediction)
            # if prediction[0] == 1:
            #     x_rects.append(x)
            #     y_rects.append(y)
            x += 2
            if x+stride >= width:
                x=0
                y+=2

        t = np.array(test_images)
        print(t.shape)
        prediction = classifier.predict(t)
        print(prediction)
        xmean=int(np.mean(x_rects))
        ymean=int(np.mean(y_rects))
        cv2.rectangle(img, (xmean, ymean), (xmean + stride, ymean + stride), (0, 255, 50), 2)
        show_img(img)
        cv2.imwrite("output/{}.png".format(filename), img)





def show_strip():
    #img = cv2.imread('input/cleaned/pos_stop/7839_stop_1324866481.avi_image26.png')
    img = cv2.imread('input/lisa/vid0/frameAnnotations-vid_cmp2.avi_annotations/stop_1323804592.avi_image7.png')
    img = cv2.imread('input/lisa/vid0/frameAnnotations-vid_cmp2.avi_annotations/pedestrian_1323804492.avi_image8.png')
    img = cv2.imread('input/lisa/vid0/frameAnnotations-vid_cmp2.avi_annotations/pedestrian_1323804492.avi_image5.png')

    # img = cv2.imread('input/lisa/vid0/frameAnnotations-vid_cmp2.avi_annotations/pedestrian_1323804463.avi_image3.png')

    saved_hist_results = getFiltersHistograms(INPUT_FILTER)

    img_in = np.copy(img)
    classifier, hog, svm_dict = saved_svm_or_fit(path='svms_cur')
    bin_classifier, hog, svm_binary_dict = saved_svm_or_fit_bin(path='svms_bin_cur')
    hog = getHogDescriptor((32,32))

    detectors = [HAARDetection('data_stop'), HAARDetection('data_diamond')]

    print(img.shape)
    possible_regions = []
    for detector in detectors:
        possible_regions.extend(detector.find_signs(img_in))
    # print(possible_regions)

    img_in = np.copy(img)
    candidate_signs = []
    candidate_borders = []

    for (x,y,w,h) in possible_regions:
        just_box = img[y:y+h,x:x+w,:]
        box_h, box_w, _ = just_box.shape
        if box_h == 0 or box_w == 0:
            continue
        # print('box shape', just_box.shape)
        # print(x,x+w, y, y+h)
        # just_box = svm_preprocessing(just_box)
        bin_res = classify(bin_classifier, hog, [just_box], svm_binary_dict)


        if bin_res[0] == 'sign':
            print('sign')
            cv2.rectangle(img_in,(x,y),(x+w,y+h),(0,100,255),2)
            candidate_signs.append(just_box)
            candidate_borders.append((x,y,w,h))
            compareHistogram(just_box, saved_hist_results)
            # candidate_signs.append(just_box)
            # candidate_signs.append(just_box)
            # candidate_signs.append(just_box)
            # candidate_signs.append(just_box)
            # font = cv2.FONT_HERSHEY_SIMPLEX
        else:
            cv2.rectangle(img_in,(x,y),(x+w,y+h),(255,255,0),2)
            # show_img(img_in)
            # cv2.putText(img, 'Sign',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

    show_img(img_in)
    # print(type(candidate_signs))
    if len(candidate_signs) > 0:
        # show_img(region[0])
        classification = classify(classifier, hog, candidate_signs, svm_dict)
        draw_classification(img, candidate_borders, classification)

    print(possible_regions[0].shape)
            # Xtest = np.array([resize(x) for x in possible_regions[0]])



def detect_signs(img):

    # faceDetection(img, classifier, hog, 'temp')

    # find regions of interest by hand
    regions = []
    # region = select_ROI(img)

    # region = find_radial_symmetry(img)
    # region = find_circles(img)
    # region = stop_sign_detection(img)
    region = HAARDetection(img)
    regions.append(region)
    regions = np.array(regions)

    return regions


def calculate_accuracy(annotation_results, calc_results):


    signs_found = 0
    exact_matches = 0
    total_signs_in_annotation = 0
    num = 0

    sign_in_annotation = defaultdict(int)
    correct_sign_in_calc = defaultdict(int)

    for annot_name, annot_vals in annotation_results.items():

        total_signs_in_annotation += len(annot_vals)
        calc_vals = calc_results.get(annot_name)
        if calc_vals and len(calc_vals) > 0:
            for annot_val in annot_vals:
                sign_in_annotation[annot_val[0]] += 1
                for calc_val in calc_vals:
                    if calc_val[0] == annot_val[0]:

                        print(annot_name, annot_vals)
                        print(calc_val)
                        signs_found += 1
                        diffs = [abs(int(calc_val[i])-int(annot_val[i])) for i in range(1, 5)]
                        match = (len([x for x in diffs if x >20]) == 0)
                        if match:
                            exact_matches += 1
                            correct_sign_in_calc[calc_val[0]] += 1
                            if annot_vals and len(annot_vals)==1:
                                break

    accuracy = 100.0 * exact_matches / total_signs_in_annotation if total_signs_in_annotation != 0 else 0.0


    for key, in_annot in sign_in_annotation.items():
        in_calc = correct_sign_in_calc.get(key, 0)
        acc = 100. * in_calc / in_annot if in_annot != 0 else 0.0
        print(key, 'shown ', in_annot, 'matched', in_calc, 'accuracy:', acc, '%')

    print('total signs found', signs_found, 'exact matches', exact_matches)
    print('total signs in annotation', total_signs_in_annotation, 'accuracy', accuracy, '%')




def filter_empty_spaces(img, possible_regions):
    img_in = np.copy(img)
    candidate_signs = []
    candidate_borders = []

    for (x,y,w,h) in possible_regions:
        just_box = img[x:x+w,y:y+h,:]
        box_h, box_w, _ = just_box.shape
        if box_h == 0 or box_w == 0:
         continue
        # print('box shape', just_box.shape)
        # print(x,x+w, y, y+h)
        # show_img(just_box)
        candidate_signs.append(just_box)
        candidate_borders.append((x,y,w,h))
        # classify(classifier, hog, [just_box])
        # candidate_signs.append(just_box)
        # candidate_signs.append(just_box)
        # candidate_signs.append(just_box)
        # candidate_signs.append(just_box)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(img_in,(x,y),(x+w,y+h),(255,255,0),2)
        # cv2.putText(img, 'Sign',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

    # show_img(img_in)
    # print(type(candidate_signs))
    print(len(candidate_signs))

    return candidate_borders, candidate_signs

def compareHistogram(img, hist_results):

    # show_img(img)
    img_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    img_hist = cv2.normalize(img_hist, None).flatten()
    hist_res = []
    for label, hist1 in hist_results.items():

        d = cv2.compareHist(img_hist, hist1, cv2.HISTCMP_CORREL)
        # if d > 0.9:
        #     print(label, ' correl ', d)
        hist_res.append(d)

    return np.array(hist_res)


def draw_classification(img_in, border, classifications):
    color_dict = {
        'stop':(255,0,255),
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    for border, txt in zip(border, classifications):
        (x,y,w,h) = border
        if txt != 'nosign':
            cv2.rectangle(img_in,(x,y),(x+w,y+h),color_dict.get(txt,(255,0,0)),2)

            cv2.putText(img_in, txt,(x+w,y+h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

    # show_img(img_in)
    return img_in

def make_imgs_brighter(img_in):
    # show_img(img_in)
    hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    hChannel = hsv[:, :, 0]
    sChannel = hsv[:, :, 1]
    vChannel = hsv[:, :, 2]

    # increase the saturation
    sChannel = sChannel * 2
    hsv = cv2.merge((hChannel, sChannel, vChannel))
    rgbVivid = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgbVivid

def detect_haar_regions(images, haar_paths):

    detectors = []
    for haar_path in haar_paths:
        detectors.append(HAARDetection(haar_path))
    possible_regions = []

    for img in images:
        img_in = np.copy(img)
        # denoising(img)
        for detector in detectors:
            possible_regions.extend(detector.find_signs(img_in))

    print(len(possible_regions))
    return possible_regions

def detect_and_classify(images, svm_paths, svm_bin_path, haar_paths, showImg=False, labels=None):

    classifiers_dict  = get_saved_svms(svm_paths)

    bin_classifier, bin_hog, svm_binary_dict = saved_svm_or_fit_bin(path=svm_bin_path)

    saved_hist_results_in = getFiltersHistograms(INPUT_FILTER_IN)
    saved_hist_results_out = getFiltersHistograms(INPUT_FILTER_OUT)

    detectors = []
    for haar_path in haar_paths:
        detectors.append(HAARDetection(haar_path))

    possible_regions = []

    output_images = []

    total_results = {}

    if not classifiers_dict:
        print('classifier does not exist, check path ')
        return

    print('num of images to process', len(images))

    num = 0
    for img in images:
        img_in = np.copy(img)
        # denoising(img)
        for detector in detectors:
            possible_regions.extend(detector.find_signs(img_in))
        # candidate_borders, candidate_signs = filter_empty_spaces(possible_regions)

        possible_signs = []
        candidate_borders = []

        # img_denoise = cv2.medianBlur(img, 3)
        img_denoise = cv2.fastNlMeansDenoisingColored(img,None,7,7,7,21)

        for (x,y,w,h) in possible_regions:
            just_box = img_denoise[y:y+h,x:x+w,:]
            box_h, box_w, _ = just_box.shape
            if box_h == 0 or box_w == 0:
                continue
            # print('box shape', just_box.shape)
            # print(x,x+w, y, y+h)

            bin_res = classify(bin_classifier, bin_hog, [just_box], svm_binary_dict)


            if bin_res[0] == 'sign':
                cv2.rectangle(img_in,(x,y),(x+w,y+h),(0,255,255),2)

                # just_box_bright = make_imgs_brighter(just_box)
                corr_res_in = compareHistogram(just_box, saved_hist_results_in)
                corr_res_out = compareHistogram(just_box, saved_hist_results_out)

                # print('out', corr_res_out)
                # print('in', corr_res_in)
                if np.any(corr_res_in > 0.50) and np.all(corr_res_out < 0.70):
                    possible_signs.append(just_box)
                    candidate_borders.append((x,y,w,h))
                else:
                    cv2.rectangle(img_in,(x,y),(x+w,y+h),(100,0,255),2)
                    # show_img(just_box)
                # candidate_signs.append(just_box)
                # candidate_signs.append(just_box)
                # candidate_signs.append(just_box)
                # candidate_signs.append(just_box)
                # font = cv2.FONT_HERSHEY_SIMPLEX
            # else:
            #     cv2.rectangle(img_in,(x,y),(x+w,y+h),(255,255,0),2)
        # show_img(img_in)
            # cv2.putText(img, 'Sign',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

        if labels:
            img_name = os.path.basename(labels[num])
        else:
            img_name = 'img_%d' % num
        num += 1

        # show_img(img_in, img_name)
        draw_img = np.copy(img)
        try:
            # show_img(region[0])
            if len(possible_signs) > 0:
                print(img_name)

                possible_signs = np.array(possible_signs)
                candidate_borders = np.array(candidate_borders)
                for classifier, hog, label_dict in classifiers_dict.values():
                    if classifier is None or hog is None:
                        print('svm classifiers not setup correctly')
                        continue

                    if len(possible_signs) > 0:
                        # print('entering again')
                        classification = classify(classifier, hog, possible_signs, label_dict)
                        classification = np.array(classification)

                        print(classification)
                        draw_img = draw_classification(draw_img, candidate_borders, classification)
                        # print(possible_signs.shape)

                        # found_signs = possible_signs[classification!='nosign']
                        found_borders = candidate_borders[classification!='nosign']
                        found_classifications = classification[classification!='nosign']

                        possible_signs = possible_signs[classification=='nosign']
                        candidate_borders = candidate_borders[classification=='nosign']

                        assert len(found_borders) == len(found_classifications), 'classifications and border do not match'

                        for clf, border in zip(found_classifications, found_borders):

                            if img_name in total_results:
                                total_results[img_name].append((clf, border[0], border[1], border[0]+border[2], border[1]+border[3]))
                            else:
                                total_results[img_name] = [(clf, border[0], border[1], border[0]+border[2], border[1]+border[3])]
                        # print(len(possible_signs))
                        # print(len(candidate_borders))
                # no signs were found by both the classifiers
                if img_name not in total_results:
                    total_results[img_name] = []

                output_images.append(draw_img)

            else:
                output_images.append(draw_img)
                if not img_name in total_results:
                    total_results[img_name] = []
        except Exception as e:
            print(e)
            print(img_name)

        if showImg:
            show_img(draw_img)
    return output_images, total_results


'''
def test_from_annotation(inpPaths, record_video=False):
    path = frame_1
    path = vid_frame_0
    annot_paths = [vid_10_csv, vid_0_csv, vid_11_csv, vid_9_csv, vid_8_csv, vid_6_csv,]

    # path = CLEAN_POS_STOP_PATH
    # signType = 'stop_'
    # signType = 'keepRight'
    signType = None
    # images = get_images_of_type(path, signType, 30)
    # img = images[0]
    # 'stop','pedestrian','signalahead', 'merge','keepright',
    images, labels, annotation_res = load_img_from_annotation(annot_paths, signType=['stop','pedestrian','signalahead', 'merge','keepright'], only_get=629)
    # print(img.shape)

    # check with one image
    # path = os.path.join(vid_frame_8, 'keepRight_1324866323.avi_image5.png')
    # im = cv2.imread(path)
    #
    # images = [im]
    # labels = {}


    # SVM_STOP_PATH = 'svms_stop'
    # SVM_PATH = 'svms_peds1'
    SVM_BIN_PATH = 'svms_bin_cur'
    OUTPUT_PATH = OUTPUT_VIDEO_PATH

    SVM_PATHS = ['svms_stop', 'svms_peds1', 'svms_signal_ahead','svms_keepright','svms_merge']
    # SVM_PATHS = ['svms_new']
    HAAR_PATHS = ['data_stop', 'data_diamond', 'data_keepright', 'data_signalAhead', 'data_resize','haarcascade']
    # HAAR_PATHS = ['data_stop']
    output_images, calc_results = detect_and_classify(images, SVM_PATHS, SVM_BIN_PATH, HAAR_PATHS, labels)

    if OUTPUT_PATH and output_images:
        imgs_to_video(output_images, OUTPUT_VIDEO_PATH)


    calculate_accuracy(annotation_res, calc_results)
'''

def main(inpPath, showImg=True, record_video=False):

    images = get_images_of_type(inpPath)


    OUTPUT_PATH = OUTPUT_VIDEO_PATH

    SVM_PATHS = [SVMS_STOP, SVMS_PEDS1, SVMS_SIGNAL_AHEAD, SVMS_KEEP_RIGHT, SVMS_MERGE]

    HAAR_PATHS = [HAAR_STOP, HAAR_DIAMOND , HAAR_KEEPRIGHT, HAAR_SIGNALAHEAD, HAAR_RESIZE, HAAR_CASCADE]

    output_images, calc_results = detect_and_classify(images, SVM_PATHS, SVMS_BIN, HAAR_PATHS, showImg=showImg)

    if record_video and OUTPUT_PATH and output_images:
        imgs_to_video(output_images, OUTPUT_VIDEO_PATH)





if __name__ == "__main__":
      # test_saved_svm()
    # test_SVM_with_cutouts()
    # detect_sign()
    main(IMAGE_INP_PATH, showImg=True, record_video=False)



