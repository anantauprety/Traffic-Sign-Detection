
import cv2


def resize(X, size=(32,32)):
    return cv2.resize(X, size)

def svm_preprocessing(X, size=(32,32)):
    # X = cv2.fastNlMeansDenoisingColored(X,None,7,7,7,21)
    return resize(X, size)

def preprocessing(X, size=(32,32)):
    # normalize all intensities to be between 0 and 1
    # X = np.array(X).astype(np.float32) / 255
    # # subtract mean
    # X = np.array([x - np.mean(x) for x in X])
    # X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    return resize(X, size)


def denoising(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.fastNlMeansDenoisingColored(img,None,7,7,7,21)
    gray = cv2.fastNlMeansDenoising(gray,None,9,7,21)

    # show_img(gray)
    # show_img(img)
    return gray

