import os

INPUT_PATH = 'input'



OUTPUT_PATH = 'output'

SVMS = 'svms'

HAAR = 'haar_trained'


IMAGE_INP_PATH = os.path.join(INPUT_PATH, 'images')
KAGGLE_PATH = os.path.join(INPUT_PATH, 'kaggle')
LISA_PATH = os.path.join(INPUT_PATH, 'lisa')


DAYSEQUENCE = os.path.join(KAGGLE_PATH, 'daySequence1')
DAYSEQUENCE_FRAME = os.path.join(DAYSEQUENCE, 'frames')

AIUA_0 = os.path.join(LISA_PATH, 'aiua120214-0')
AIUA_1 = os.path.join(LISA_PATH, 'aiua120214-1')
AIUA_2 = os.path.join(LISA_PATH, 'aiua120214-2')
AIUA_3 = os.path.join(LISA_PATH, 'aiua120306-0')

VID_0 = os.path.join(LISA_PATH, 'vid0')
VID_1 = os.path.join(LISA_PATH, 'vid1')
VID_2 = os.path.join(LISA_PATH, 'vid2')
VID_3 = os.path.join(LISA_PATH, 'vid3')
VID_4 = os.path.join(LISA_PATH, 'vid4')
VID_5 = os.path.join(LISA_PATH, 'vid5')
VID_6 = os.path.join(LISA_PATH, 'vid6')
VID_7 = os.path.join(LISA_PATH, 'vid7')
VID_8 = os.path.join(LISA_PATH, 'vid8')
VID_9 = os.path.join(LISA_PATH, 'vid9')
VID_10 = os.path.join(LISA_PATH, 'vid10')
VID_11 = os.path.join(LISA_PATH, 'vid11')

vid_frame_0 = os.path.join(VID_0,
                'frameAnnotations-vid_cmp2.avi_annotations')

vid_frame_1 = os.path.join(VID_1,
                'frameAnnotations-vid_cmp1.avi_annotations')

vid_frame_2 = os.path.join(VID_2,
                'frameAnnotations-vid_cmp2.avi_annotations')

vid_frame_3 = os.path.join(VID_3,
                'frameAnnotations-vid_cmp2.avi_annotations')

vid_frame_4 = os.path.join(VID_4,
                'frameAnnotations-vid_cmp2.avi_annotations')

vid_frame_5 = os.path.join(VID_5,
                'frameAnnotations-vid_cmp2.avi_annotations')


vid_frame_6 = os.path.join(VID_6,
                'frameAnnotations-MVI_0071.MOV_annotations')

vid_frame_7 = os.path.join(VID_7,
                'frameAnnotations-MVI_0119.MOV_annotations')

vid_frame_8 = os.path.join(VID_8,
                'frameAnnotations-MVI_0120.MOV_annotations')
vid_frame_9 = os.path.join(VID_9,
                'frameAnnotations-MVI_0121.MOV_annotations')
vid_frame_10 = os.path.join(VID_10,
                'frameAnnotations-MVI_0122.MOV_annotations')
vid_frame_11 = os.path.join(VID_11,
                'frameAnnotations-MVI_0123.MOV_annotations')


frame_0 = os.path.join(AIUA_0,
                'frameAnnotations-DataLog02142012_external_camera.avi_annotations')

frame_1 = os.path.join(AIUA_1,
                'frameAnnotations-DataLog02142012_001_external_camera.avi_annotations')

frame_2 = os.path.join(AIUA_2,
                       'frameAnnotations-DataLog02142012_002_external_camera.avi_annotations')
frame_3 = os.path.join(AIUA_3,
                       'frameAnnotations-DataLog02142012_002_external_camera.avi_annotations')

vid_0_csv = os.path.join(vid_frame_0, 'frameAnnotations.csv')
vid_1_csv = os.path.join(vid_frame_1, 'frameAnnotations.csv')
vid_2_csv = os.path.join(vid_frame_2, 'frameAnnotations.csv')
vid_3_csv = os.path.join(vid_frame_3, 'frameAnnotations.csv')
vid_4_csv = os.path.join(vid_frame_4, 'frameAnnotations.csv')
vid_5_csv = os.path.join(vid_frame_5, 'frameAnnotations.csv')
vid_6_csv = os.path.join(vid_frame_6, 'frameAnnotations.csv')
vid_7_csv = os.path.join(vid_frame_7, 'frameAnnotations.csv')
vid_8_csv = os.path.join(vid_frame_8, 'frameAnnotations.csv')
vid_9_csv = os.path.join(vid_frame_9, 'frameAnnotations.csv')
vid_10_csv = os.path.join(vid_frame_10, 'frameAnnotations.csv')
vid_11_csv = os.path.join(vid_frame_11, 'frameAnnotations.csv')

annotations_0 = os.path.join(frame_0, 'annotations')
annotations_1 = os.path.join(frame_1, 'annotations')
annotations_2 = os.path.join(frame_2, 'annotations')
annotations_3 = os.path.join(frame_3, 'annotations')

vid_annotations_0 = os.path.join(vid_frame_0, 'annotations')
vid_annotations_1 = os.path.join(vid_frame_1, 'annotations')
vid_annotations_2 = os.path.join(vid_frame_2, 'annotations')
vid_annotations_3 = os.path.join(vid_frame_3, 'annotations')
vid_annotations_4 = os.path.join(vid_frame_4, 'annotations')
vid_annotations_5 = os.path.join(vid_frame_5, 'annotations')
vid_annotations_6 = os.path.join(vid_frame_6, 'annotations')
vid_annotations_7 = os.path.join(vid_frame_7, 'annotations')
vid_annotations_8 = os.path.join(vid_frame_8, 'annotations')
vid_annotations_9 = os.path.join(vid_frame_9, 'annotations')
vid_annotations_10 = os.path.join(vid_frame_10, 'annotations')
vid_annotations_11 = os.path.join(vid_frame_11, 'annotations')

NEGATIVE_PATH = os.path.join(LISA_PATH, 'negatives')
NEGATIVE_PICS = os.path.join(NEGATIVE_PATH, 'negativePics')
NEGATIVE_PICS_COPY = os.path.join(NEGATIVE_PATH, 'negativePics_copy')

TOOLS_PATH = os.path.join(LISA_PATH, 'tools')
TOOLS_ANNOTATIONS = os.path.join(TOOLS_PATH, 'annotations')
TOOLS_ANNOTATIONS_NEG = os.path.join(TOOLS_PATH, 'annotations_negative')

CLEANED_PATH = os.path.join(INPUT_PATH, 'cleaned')
CLEAN_POS_PATH = os.path.join(CLEANED_PATH, 'pos')
CLEAN_POS_UPDATE_PATH = os.path.join(CLEANED_PATH, 'pos_upd')
CLEAN_POS_UPDATE_GREY_PATH = os.path.join(CLEANED_PATH, 'pos_upd_grey')
CLEAN_POS_STOP_PATH = os.path.join(CLEANED_PATH, 'pos_stop')

CLEAN_NEG_PATH = os.path.join(CLEANED_PATH, 'neg')
CLEAN_NEG_REN_PATH = os.path.join(CLEANED_PATH, 'neg_rename')

CLEAN_POS_RESIZE_PATH = os.path.join(CLEANED_PATH, 'pos_stop_resize')
CLEAN_POS_GREY_PATH = os.path.join(CLEANED_PATH, 'pos_grey')

MERGED_POS_PATH = os.path.join(INPUT_PATH, 'merged_pos')
MERGED_POS_GREY_PATH =  os.path.join(INPUT_PATH, 'merged_pos_grey')

OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_PATH, 'video')

INPUT_FILTER_IN = os.path.join(INPUT_PATH, 'filter_in')
INPUT_FILTER_OUT= os.path.join(INPUT_PATH, 'filter_out')

SVMS_STOP = os.path.join(SVMS, 'svms_stop')
SVMS_PEDS1 = os.path.join(SVMS, 'svms_peds1')
SVMS_SIGNAL_AHEAD = os.path.join(SVMS, 'svms_signal_ahead')
SVMS_KEEP_RIGHT = os.path.join(SVMS, 'svms_keepright')
SVMS_MERGE = os.path.join(SVMS, 'svms_merge')
SVMS_BIN = os.path.join(SVMS, 'svms_bin_cur')

HAAR_DIAMOND = os.path.join(HAAR, 'diamond_cascade.xml')
HAAR_CASCADE = os.path.join(HAAR, 'haarcascade.xml')
HAAR_KEEPRIGHT = os.path.join(HAAR, 'keepright_cascade.xml')
HAAR_RESIZE = os.path.join(HAAR, 'resize_cascade.xml')
HAAR_SIGNALAHEAD = os.path.join(HAAR, 'signalahead_cascade.xml')
HAAR_SQUARE = os.path.join(HAAR, 'square_cascade.xml')
HAAR_STOP = os.path.join(HAAR, 'stop_cascade.xml')
