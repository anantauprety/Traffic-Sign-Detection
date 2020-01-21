import cv2
import numpy as np
from pprint import pprint
from classifier.svm_classifier import MultiClassSVM, reload_classifier
from preprocess import  resize, svm_preprocessing, denoising


from path_collection import  *
from utils import load_images_from_folders, getShape, get_label


# goes with svms_stop
TRAFFIC_SIGN_LABEL_DICT_PED ={
    'pedestrian':0,
    'nosign':1,
}

TRAFFIC_SIGN_LABEL_DICT_SIGNAL_AHEAD ={
    'signalahead':0,
    'nosign':1,
}

TRAFFIC_SIGN_LABEL_DICT_STOP ={
    'stop':0,
    'nosign':1,
}

SIGN_NOSIGN_DICT ={
   'sign':0,
   'else':1,
}

TRAFFIC_SIGN_LABEL_DICT_KEEPRIGHT = {
    'keepright':0,
    'nosign':1,
}

TRAFFIC_SIGN_LABEL_DICT = {
    'stop':0,
    'turnright':1,
    'signalahead':2,
    'pedestrian':3,
    'merge':4,
    'nosign':5,
}

TRAFFIC_SIGN_LABEL_DICT_STOP_AHEAD = {
    'stopahead':0,
    'nosign':1,
}

TRAFFIC_SIGN_LABEL_DICT_MERGE = {
    'merge':0,
    'nosign':1,
}


HOG_SIZE = (32,32)

def get_svm_training_paths():
    # paths = [annotations_0, annotations_1, annotations_2, annotations_3,
    #             vid_annotations_0, vid_annotations_1, vid_annotations_2,
    #             vid_annotations_3, vid_annotations_4, vid_annotations_5,
    #             vid_annotations_6, MERGED_POS_PATH, TOOLS_ANNOTATIONS_NEG]
    paths = [annotations_0,vid_annotations_0,MERGED_POS_PATH]
    return paths

def get_svm_binary_training_path():
    paths = [MERGED_POS_PATH, CLEAN_POS_UPDATE_PATH, TOOLS_ANNOTATIONS_NEG]
    return paths


def getHogDescriptor(small_size=HOG_SIZE):
    block_size = (small_size[0] // 2, small_size[1] // 2)
    block_stride = (small_size[0] // 4, small_size[1] // 4)
    cell_size = block_stride
    num_bins = 9
    hog = cv2.HOGDescriptor(small_size, block_size, block_stride, cell_size, num_bins)
    return hog


def allSignFilter(path, labelPath, nosign_count):

    label = get_label(labelPath)

    # if 'neg' in path and labelPath.count('_') < 4:
    #     if nosign_count > 5000:
    #       return False, 'nosign', len(TRAFFIC_SIGN_LABEL_DICT)-1
    #     return True, 'nosign', len(TRAFFIC_SIGN_LABEL_DICT)-1

    label_num = TRAFFIC_SIGN_LABEL_DICT.get(label, len(TRAFFIC_SIGN_LABEL_DICT))
    if label_num > len(TRAFFIC_SIGN_LABEL_DICT)-1:
        if nosign_count > 2000:
            False, label, len(TRAFFIC_SIGN_LABEL_DICT)-1
        return True, label, len(TRAFFIC_SIGN_LABEL_DICT)-1

    # if label_num == len(TRAFFIC_SIGN_LABEL_DICT)-1:
    #     nosign_count += 1
    #     if nosign_count > 4000:
    #         False
    # print(label, label_num)
    return True, label, label_num



def sign_nosignFilter(path, labelPath, nosign_count):

    if 'neg' in path and labelPath.count('_') < 4:
        if nosign_count > 5000:
          return False, 'else', 1
        return True, 'else', 1
    else:
        if getShape(labelPath) in ['circle','diamond']:
            return True, 'sign', 0
    return False, 'else', 1


def train_classifier_and_save(Xtrain, ytrain, label_dict, params=None, hog_size=HOG_SIZE, path='svms'):

    hog = getHogDescriptor(hog_size)
    #
    # Xtrain = np.array([hog.compute(x).flatten() for x in Xtrain])

    num_of_classifiers = len(label_dict)
    classifier = MultiClassSVM(num_of_classifiers)
    classifier.fit_and_save(Xtrain, ytrain, params, path)
    return classifier, hog



def get_trained_classifier(path='svms',hog_size=HOG_SIZE):
    hog = getHogDescriptor(hog_size)
    classifier = reload_classifier(path)
    if classifier:
        print('found saved classifier')
        return classifier, hog

    return None, None





def load_hog_img_array(paths, only_get=None, hog_size=HOG_SIZE, dictToUse=TRAFFIC_SIGN_LABEL_DICT, labelFilter=allSignFilter):
    labels = []
    processed_images = []

    num = 0
    label_nums = []
    label_num_counter_dict = {}
    nosign_count = 0
    hog = getHogDescriptor(hog_size)
    # Xtrain =
    label_counter_dict = {}
    for path in paths:
        for labelPath in os.listdir(path):
            imagePath = os.path.join(path, labelPath)
            # print(imagePath)
            if '.png' in imagePath:

                # print(img.shape)


                toAdd, label, label_num = labelFilter(path, labelPath, nosign_count)
                if toAdd:
                    img = cv2.imread(imagePath)


                    img = svm_preprocessing(img)

                    labels.append(label)
                    label_nums.append(label_num)
                    # print('computing hog')
                    hg = hog.compute(img).flatten()
                    processed_images.append(hg)

                    if label in label_counter_dict.keys():
                        label_counter_dict[label] += 1
                    else:
                        label_counter_dict[label] = 1

                    if label_num in label_num_counter_dict.keys():
                        label_num_counter_dict[label_num] += 1
                    else:
                        label_num_counter_dict[label_num] = 1

                    if len(dictToUse)-1 == label_num:
                        nosign_count += 1

                    num += 1

                    if only_get and num > only_get:
                        break

        print('total data processed', len(processed_images), 'total labels processed', len(labels), 'for ', path)
    pprint(sorted(label_counter_dict.items(), key=lambda kv:kv[1]))
    pprint(sorted(label_num_counter_dict.items(), key=lambda kv:kv[1]))
    return np.array(processed_images), np.array(label_nums), np.array(labels)



def load_image_and_create_labels(paths, only_get=None):
    images, label_paths = load_images_from_folders(paths, only_get)

    processed_images = []
    labels = []
    label_nums = []
    num = 0
    for img, labelPath  in zip(images, label_paths):
        img = svm_preprocessing(img)
        label = 'nosign' if 'nosign' in labelPath else labelPath.split('_')[1]
        labels.append(label)
        label_num = TRAFFIC_SIGN_LABEL_DICT.get(label, len(TRAFFIC_SIGN_LABEL_DICT))
        label_nums.append(label_num)
        # print(label_num)
        # if num < 5:
        #     show_img(img, label)
        num += 1
        processed_images.append(img)
        if only_get and num > only_get:
            break

    print('total data processed', len(images), 'total labels processed', len(labels))
    return np.array(processed_images), np.array(label_nums), np.array(labels)


def split_data(X, y, p):
    total = X.shape[0]
    to_select = int(np.round(p * total))
    # print to_select
    rand_index = np.random.permutation(total)
    train_index = rand_index[:to_select]
    test_index = rand_index[to_select:]
    Xtrain = X[train_index,:]
    ytrain = y[train_index]
    Xtest = X[test_index,:]
    ytest = y[test_index]
    return Xtrain, ytrain, Xtest, ytest



def train_SVM_Classifier():
    # paths = [CLEAN_POS_UPDATE_PATH]
    paths = get_svm_training_paths()
    Xtrain, ytrain, label_names = load_hog_img_array(paths, dictToUse=TRAFFIC_SIGN_LABEL_DICT)

    hog = getHogDescriptor()
    #
    # Xtrain = np.array([hog.compute(x).flatten() for x in Xtrain])

    num_of_classifiers = len(TRAFFIC_SIGN_LABEL_DICT) + 1
    classifier = MultiClassSVM(num_of_classifiers)
    classifier.fit(Xtrain, ytrain)
    return classifier, hog


def test_SVM_with_cutouts(p=0.8):
    paths = get_svm_training_paths()
    X, y, label_names = load_hog_img_array(paths, dictToUse=TRAFFIC_SIGN_LABEL_DICT)
    Xtrain, ytrain, Xtest, ytest = split_data(X, y, p)

    # hog = getHogDescriptor()
    # Xtrain = np.array([hog.compute(x).flatten() for x in Xtrain])
    # Xtest = np.array([hog.compute(x).flatten() for x in Xtest])

    pred_dict = dict(zip(TRAFFIC_SIGN_LABEL_DICT.values(), TRAFFIC_SIGN_LABEL_DICT.keys()))
    print('unique signs', set(label_names))
    print('unique', set(y))
    print('train', Xtrain.shape, ytrain.shape)
    print('test', Xtest.shape, ytest.shape)
    for cl in set(y):
        print('class', pred_dict.get(cl))
        print('train pos ',len(ytrain[ytrain==cl]), 'train neg ', len(ytrain[ytrain!=cl]))
        print('test pos ',len(ytest[ytest==cl]), 'test neg ', len(ytest[ytest!=cl]))


    num_of_classifiers = len(TRAFFIC_SIGN_LABEL_DICT)
    classifier = MultiClassSVM(num_of_classifiers)
    classifier.fit(Xtrain, ytrain)

    classifier.evaluate(Xtest, ytest)

def saved_svm_or_fit(path='svms', train=False):
    classifier, hog =  get_trained_classifier(path)
    if not classifier and train:
        print('starting to train')
        paths = get_svm_training_paths()
        Xtrain, ytrain, label_names = load_image_and_create_labels(paths)

        classifier, hog = train_classifier_and_save(Xtrain, ytrain, TRAFFIC_SIGN_LABEL_DICT, params=None, hog_size=HOG_SIZE, path='svms_new')

    return classifier, hog, TRAFFIC_SIGN_LABEL_DICT

def saved_svm_or_fit_bin(path='svms_bin_new', train=False):
    classifier, hog =  get_trained_classifier(path)
    if not classifier and train:
        print('starting to train')
        paths = get_svm_training_paths()
        Xtrain, ytrain, label_names = load_image_and_create_labels(paths)

        classifier, hog = train_classifier_and_save(Xtrain, ytrain, TRAFFIC_SIGN_LABEL_DICT, params=None, hog_size=HOG_SIZE, path='svms_new')

    return classifier, hog, SIGN_NOSIGN_DICT


def test_saved_svm(only_get=None, p=0.8):

    paths = get_svm_training_paths()
    X, y, label_names = load_hog_img_array(paths, only_get, dictToUse=TRAFFIC_SIGN_LABEL_DICT)
    Xtrain, ytrain, Xtest, ytest = split_data(X, y, p)

    SVM_SAVE_PATH = 'svms_new'
    # # want to make sure, whether the saving and updating is the same
    init_classifier, init_hog = train_classifier_and_save(Xtrain, ytrain, TRAFFIC_SIGN_LABEL_DICT, params=None, hog_size=HOG_SIZE, path=SVM_SAVE_PATH)
    # get the saved one
    saved_classifier, saved_hog =  get_trained_classifier(path=SVM_SAVE_PATH)
    pred_dict = dict(zip(TRAFFIC_SIGN_LABEL_DICT.values(), TRAFFIC_SIGN_LABEL_DICT.keys()))

    print('unique signs', set(label_names))
    print('unique', set(y))
    print('train', Xtrain.shape, ytrain.shape)
    print('test', Xtest.shape, ytest.shape)
    for cl in set(y):
        print('class', pred_dict.get(cl))
        print('train pos ',len(ytrain[ytrain==cl]), 'train neg ', len(ytrain[ytrain!=cl]))
        print('test pos ',len(ytest[ytest==cl]), 'test neg ', len(ytest[ytest!=cl]))

    for classifier, hog in [(init_classifier, init_hog), (saved_classifier, saved_hog)]:
        # test = np.array([hog.compute(x).flatten() for x in Xtest])

        classifier.evaluate(Xtest, ytest)



def classify(classifier, hog, Xtest, labelDict=TRAFFIC_SIGN_LABEL_DICT):
    Xtest = np.array([svm_preprocessing(x) for x in Xtest])
    # print('test shape', Xtest.shape)
    Xtest = np.array([hog.compute(x).flatten() for x in Xtest])
    # print('test shape', Xtest.shape)
    predictions = classifier.predict(Xtest)
    # print(predictions)
    pred_dict = dict(zip(labelDict.values(), labelDict.keys()))
    results = []
    for predNum in predictions:
        r = pred_dict.get(predNum, 'nosign')
        # print(r)
        results.append(r)
    return results




def test_binary_svm_cutouts(p=0.8):
    paths = get_svm_binary_training_path()
    X, y, label_names = load_hog_img_array(paths, only_get=None, hog_size=HOG_SIZE, labelFilter=sign_nosignFilter, dictToUse=SIGN_NOSIGN_DICT)
    Xtrain, ytrain, Xtest, ytest = split_data(X, y, p)

    # hog = getHogDescriptor()
    # Xtrain = np.array([hog.compute(x).flatten() for x in Xtrain])
    # Xtest = np.array([hog.compute(x).flatten() for x in Xtest])

    pred_dict = dict(zip(SIGN_NOSIGN_DICT.values(), SIGN_NOSIGN_DICT.keys()))
    print('unique signs', set(label_names))
    print('unique', set(y))
    print('train', Xtrain.shape, ytrain.shape)
    print('test', Xtest.shape, ytest.shape)
    for cl in set(y):
        print('class', pred_dict.get(cl))
        print('train pos ',len(ytrain[ytrain==cl]), 'train neg ', len(ytrain[ytrain!=cl]))
        print('test pos ',len(ytest[ytest==cl]), 'test neg ', len(ytest[ytest!=cl]))


    num_of_classifiers = len(SIGN_NOSIGN_DICT)
    classifier = MultiClassSVM(num_of_classifiers)
    classifier.fit(Xtrain, ytrain)

    classifier.evaluate(Xtest, ytest)

def test_saved_svm_bin(only_get=None, p=0.8):


    paths = get_svm_binary_training_path()
    X, y, label_names = load_hog_img_array(paths, only_get=only_get, hog_size=HOG_SIZE, labelFilter=sign_nosignFilter,dictToUse=SIGN_NOSIGN_DICT)
    Xtrain, ytrain, Xtest, ytest = split_data(X, y, p)

    SVM_SAVE_PATH = 'svms_bin_new'
    # # want to make sure, whether the saving and updating is the same
    init_classifier, init_hog = train_classifier_and_save(Xtrain, ytrain, SIGN_NOSIGN_DICT, params=None, hog_size=HOG_SIZE, path=SVM_SAVE_PATH)
    # get the saved one
    saved_classifier, saved_hog =  get_trained_classifier(path=SVM_SAVE_PATH)
    pred_dict = dict(zip(SIGN_NOSIGN_DICT.values(), SIGN_NOSIGN_DICT.keys()))

    print('unique signs', set(label_names))
    print('unique', set(y))
    print('train', Xtrain.shape, ytrain.shape)
    print('test', Xtest.shape, ytest.shape)
    for cl in set(y):
        print('class', pred_dict.get(cl))
        print('train pos ',len(ytrain[ytrain==cl]), 'train neg ', len(ytrain[ytrain!=cl]))
        print('test pos ',len(ytest[ytest==cl]), 'test neg ', len(ytest[ytest!=cl]))

    for classifier, hog in [(init_classifier, init_hog), (saved_classifier, saved_hog)]:
        # test = np.array([hog.compute(x).flatten() for x in Xtest])

        classifier.evaluate(Xtest, ytest)




def get_saved_svms(svm_paths):
    svms_dict = {}

    for svm_path in svm_paths:
        if 'stop' in svm_path:
            classifier, hog, svm_binary_dict = saved_svm_or_fit(path=svm_path)
            svm_binary_dict = TRAFFIC_SIGN_LABEL_DICT_STOP
            svms_dict[svm_path] = (classifier, hog, svm_binary_dict)
        elif 'ped' in svm_path:
            classifier, hog, svm_binary_dict = saved_svm_or_fit(path=svm_path)
            svm_binary_dict = TRAFFIC_SIGN_LABEL_DICT_PED
            svms_dict[svm_path] = (classifier, hog, svm_binary_dict)
        elif 'signal' in svm_path:
            classifier, hog, svm_binary_dict = saved_svm_or_fit(path=svm_path)
            svm_binary_dict = TRAFFIC_SIGN_LABEL_DICT_SIGNAL_AHEAD
            svms_dict[svm_path] = (classifier, hog, svm_binary_dict)
        elif 'keepright' in svm_path:
            classifier, hog, svm_binary_dict = saved_svm_or_fit(path=svm_path)
            svm_binary_dict = TRAFFIC_SIGN_LABEL_DICT_KEEPRIGHT
            svms_dict[svm_path] = (classifier, hog, svm_binary_dict)
        elif 'merge' in svm_path:
            classifier, hog, svm_binary_dict = saved_svm_or_fit(path=svm_path)
            svm_binary_dict = TRAFFIC_SIGN_LABEL_DICT_MERGE
            svms_dict[svm_path] = (classifier, hog, svm_binary_dict)
        else:
            classifier, hog, svm_binary_dict = saved_svm_or_fit(path=svm_path)
            svms_dict[svm_path] = (classifier, hog, svm_binary_dict)
    return svms_dict


if __name__ == "__main__":
    test_SVM_with_cutouts()
    # saved_svm_or_fit()
    # test_binary_svm_cutouts()
    # test_saved_svm_bin()
    # test_saved_svm()

