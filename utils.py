
import cv2
import numpy as np
import image_slicer
from path_collection import *
from pprint import pprint

import re

DEBUG = True

num_re = re.compile('^[0-9]')


TRAFFIC_SIGN_SHAPE_DICT = {'noleftturn': 'circle',
                           'stop': 'circle', 'curveleft': 'diamond',
                           'dip': 'diamond', 'laneends': 'diamond',
                           'leftturn': 'diamond', 'merge': 'diamond',
                           'pedestrian': 'diamond',
                           'pedestriancrossing': 'diamond',
                           'roundabout': 'diamond',
                           'signalahead': 'diamond',
                           'stopahead': 'diamond',
                           'thrumergeleft': 'diamond',
                           'thrumergeright': 'diamond', 'turnright': 'diamond',
                           'keepright': 'square', 'rampspeedadvisory': 'square', 'rightlatemustturn': 'square', 'schoolspeedlimit': 'square', 'speedlimit': 'square', 'truckspeedlimit': 'square'}


def check_folder(paths):

 for path in paths:
        for labelPath in os.listdir(path):
            imagePath = os.path.join(path, labelPath)
            # print(imagePath)
            if '.png' not in imagePath:
                print(imagePath)

def show_img(img_in, label='blah'):
    if DEBUG:
        cv2.imshow(label, img_in)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def load_images_from_folders(paths, only_get=None):
    images = []
    label_names = []
    num = 0
    for path in paths:
        for labelPath in os.listdir(path):
            imagePath = os.path.join(path, labelPath)
            # print(imagePath)
            if '.png' in imagePath:
                img = cv2.imread(imagePath)
                # print(img.shape)
                label_names.append(labelPath)
                images.append(img)
                # show_img(img)
                num += 1
                if only_get and num > only_get:
                    break

    print('loaded %d image and %d paths' % (len(images), len(label_names)))
    return images, label_names

def load_images_from_paths(absPaths):
    images = []
    label_names = []
    for imgPath in absPaths:
        img = cv2.imread(imgPath)
        images.append(img)
        label_names.append(imgPath)

    print('loaded %d image and %d paths' % (len(images), len(label_names)))
    return images, label_names


def cut_my_pics_into_pieces(path):
    outputPath = os.path.join(path, 'pieces')


    # load_images([path])
    for labelPath in os.listdir(path):
            imagePath = os.path.join(path, labelPath)

            # img = cv2.imread(imagePath)
            # print(img.shape)
            # return

            # if '.png' in imagePath and labelPath.count('_') > 2:
            tiles = image_slicer.slice(imagePath, 20, save=True)
            print(imagePath)
            # image_slicer.save_tiles(tiles, directory=outputPath, prefix='slice', format='jpg')
                # for tile in tiles:
                #     print(type(tile))
                #     print(tile)
                #     tile_img = np.array(tile)
                #     if tile_img.dtype == bool:
                #         print('is bool')
                #         tile_img = tile_img.astype(np.uint8) * 255
                #     tile_img = cv2.convert(tile_img, cv2.CV_8SC1, cv2.CV_16SC2)
                #     print(tile_img.shape)
                #     # tile_img = cv2.cvtColor(np.array(tile), cv2.COLOR_RGB2BGR)
                #     show_img(tile_img)
    print(outputPath)
    print('done')

def get_label(labelPath):
    if 'nosign' in labelPath:
        label = 'nosign'
    else:
        name_or_num = labelPath.split('_')
        if num_re.match(name_or_num[0]):
            label = name_or_num[1]
        else:
            label = name_or_num[0]

        if 'speedLimit' in label:
            label ='speedlimit'
        if 'pedestrian' in label:
            label = 'pedestrian'

    return label.lower()

def print_stat(paths, only_get=None):
    label_dict = {}
    num = 0
    txts = []
    for path in paths:
        for labelPath in os.listdir(path):
            # imagePath = os.path.join(path, labelPath)
            if '.png' in labelPath:
                label = get_label(labelPath)

                if label in label_dict.keys():
                    label_dict[label] += 1
                else:
                    label_dict[label] = 1

    dict_sort = sorted(label_dict.items(), key=lambda kv:kv[1])
    for key, val in dict_sort:
        txts.append('%s,%d' % (key, val))

    print('\n'.join(txts))


def readGSR():
    path = '/home/pablo/Downloads/GTSRB/Final_Training/Images/00038'
    # print(LISA_PATH)
    for labelPath in os.listdir(path):
        print(labelPath)
    # print_stat([CLEAN_POS_UPDATE_GREY_PATH])
        imagePath = os.path.join(path, labelPath)
        print(imagePath)
        # im = cv2.imread('/home/pablo/Downloads/GTSRB/Final_Training/Images/00014/00000_00001.ppm')
        img = cv2.imread(imagePath)
        print(img.shape)
        show_img(img)


def removeBackGround(path):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()


    for labelPath in os.listdir(path):
        imagePath = os.path.join(path, labelPath)
        if '.png' in labelPath :
            img = cv2.imread(imagePath)

            fgmask = fgbg.apply(img)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            cv2.imshow('frame',fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


def getShape(filePath):
    name_or_num = filePath.split('_')
    if num_re.match(name_or_num[0]):
        name = name_or_num[1]
    else:
        name = name_or_num[0]

    name = name.lower()
    if 'speedLimit' in name:
        name ='speedlimit'
    if 'pedestrian' in name:
        name = 'pedestrian'
    shape = TRAFFIC_SIGN_SHAPE_DICT.get(name,'blah')
    return shape

'''
video_file = "ps3-4-b.mp4"
    frame_ids = [97, 407, 435]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-a", 4, False)
    
    mp4_video_writer(out_path, (w, h), fps)
    '''


def mp4_video_writer(filename, frame_size=(640,480), fps=3):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    # fourcc = cv2.cv.CV_FOURCC(*'MP4V')

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def test_video_writer(path):
    fileName = get_video_fileName(path)
    video_out = mp4_video_writer(fileName)


    cnt = 0
    image_info = load_images_from_folders([vid_frame_0], 20)
    images = image_info[0]
    print(type(image_info))
    print(len(image_info))
    for img in images:
        # show_img(img)

        img = cv2.resize(img, (640,480))
        print(img.shape)
        video_out.write(img)
        cnt += 1
        if cnt > 20:
            break

    #
    video_out.release()


def get_video_fileName(path):
    fileName = 'traffic_sign_video'

    num = len(os.listdir(path))
    fileName = '%s_%d.avi' % (fileName,num)
    fullPath = os.path.join(path, fileName)
    print('video writing to', fullPath)
    return fullPath


def read_annotationfile(paths, signType=None, only_get=20):


    res_paths = []

    act_sign = []
    sign_dict = {}
    num = 0
    for path in paths:
        if '.csv' not in path:
            print('no csv file given')
            return None, None



        csv = open(os.path.abspath(path), 'r')
        csv.readline() # Discard the header-line.
        csv = csv.readlines()
        # csv.sort()

        basePath = os.path.dirname(path)

        print(basePath)

        for line in csv:
            fields = line.split(";")

            imgName = fields[0]
            sign = fields[1]
            upper_left_x = fields[2]
            upper_left_y = fields[3]
            lower_right_x = fields[4]
            lower_right_y = fields[5]

            # print(imgName)
            # print(sign)
            # print(signType)

            if signType:
                labelName = get_label(sign)
                if labelName in signType:
                    print(imgName)

                    fullPath = os.path.join(basePath, imgName)
                    res_paths.append(fullPath)
                    if imgName in sign_dict:
                        sign_dict[imgName].append((labelName, upper_left_x, upper_left_y, lower_right_x, lower_right_y))
                    else:
                        sign_dict[imgName] = [(labelName, upper_left_x, upper_left_y, lower_right_x, lower_right_y)]
                    num += 1
                    if only_get and num >= only_get:
                        break
            else:
                print(imgName)

                fullPath = os.path.join(basePath, imgName)
                res_paths.append(fullPath)

                sign_dict[imgName] = (sign, upper_left_x, upper_left_y, lower_right_x, lower_right_y)

                num += 1
                if only_get and num >= only_get:
                    break

    return res_paths, sign_dict


def load_img_from_annotation(annot_paths,signType=None, only_get=None):


    paths, annotation_res = read_annotationfile(annot_paths, signType, only_get)
    images, labels = load_images_from_paths(paths)

    # return images, labels
    # imgs_to_video(images)
    return images, labels, annotation_res

def imgs_to_video(images, outputPath=OUTPUT_VIDEO_PATH):
    fileName = get_video_fileName(outputPath)
    video_out = mp4_video_writer(fileName, frame_size=(640,480))

    cnt = 0
    for img in images:
        # show_img(img)
        # print(img.shape)
        img = cv2.resize(img, (640,480))
        video_out.write(img)
        cnt += 1
        # if cnt > 20:
        #     break
    #
    video_out.release()



def measure_match(truthVAlues, predictions):
    pass


def getFiltersHistograms(filter_path):
    hist_results = {}
    for labelPath in os.listdir(filter_path):
        imagePath = os.path.join(filter_path, labelPath)
        if '.png' in labelPath or '.jpg' in labelPath:

            img = cv2.imread(imagePath)
            # show_img(img, labelPath)
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

            hist = cv2.normalize(hist,None).flatten()

            hist_results[labelPath] = hist

    return hist_results

if __name__ == "__main__":
    # print(vid_annotations_0)
    cut_my_pics_into_pieces(vid_annotations_2)
    # print_stat([annotations_0, annotations_1, annotations_2, annotations_3,
    #             vid_annotations_0, vid_annotations_1, vid_annotations_2,
    #             vid_annotations_3, vid_annotations_4, vid_annotations_5,
    #             vid_annotations_6,MERGED_POS_PATH])
    # print_stat([MERGED_POS_PATH])
    # check_folder([vid_frame_11])

    # removeBackGround(vid_annotations_1)
    # print(get_video_fileName(OUTPUT_VIDEO_PATH))
    # test_video_writer(OUTPUT_VIDEO_PATH)
    # pprint(read_annotationfile(vid_0_csv))
    # load_img_from_annotation(vid_0_csv)

    # path = os.path.join(vid_frame_8, 'keepRight_1324866323.avi_image5.png')
    # im = cv2.imread(path)
    # show_img(im)

