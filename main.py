import math

import cv2
import numpy as np
import scipy.signal as signal
from scipy.sparse.linalg.isolve.lsqr import eps


def get_corners(_frame):
    _X = np.zeros(0)
    _Y = np.zeros(0)
    gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

    blockSize = 2
    apertureSize = 3
    k = 0.04
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize, apertureSize, k)

    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    for _i in range(dst_norm.shape[0]):
        for _j in range(dst_norm.shape[1]):
            if int(dst_norm[_i, _j]) > 200:
                _X = np.append(_X, int(_j))
                _Y = np.append(_Y, int(_i))
    return _X, _Y


def lucas_kanade(_frame, _last_frame, _X, _Y, _n):
    _orig_frame = _frame

    _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    _last_frame = cv2.cvtColor(_last_frame, cv2.COLOR_BGR2GRAY)

    kernel_t = np.array([[1., 1.], [1., 1.]])
    kernel_sum = np.ones((k_size, k_size))

    _grad_x = cv2.Sobel(_frame, cv2.CV_64F, 1, 0, ksize=3)
    _grad_y = cv2.Sobel(_frame, cv2.CV_64F, 0, 1, ksize=3)
    _grad_t = signal.convolve2d(_last_frame, kernel_t, boundary='symm', mode='same') + signal.convolve2d(_frame,
                                                                                                         -kernel_t,
                                                                                                         boundary='symm',
                                                                                                         mode='same')

    k_w = k_size
    k_h = k_size

    I_x_k = np.zeros((k_size, k_size))
    I_y_k = np.zeros((k_size, k_size))
    I_t_k = np.zeros((k_size, k_size))

    # Go through each tracking point and calculate velocity using the Lucas-Kanade method
    for _i in range(0, _n):
        _x = _X[_i]
        for _j in range(0, _n):
            _y = _Y[_j]
            o_x = k_w - _x
            o_y = k_h - _y
            for _yi in range(0, k_h):
                I_x = _grad_x[o_y + _yi][o_x - math.floor(k_w / 2): o_x + math.floor(k_w / 2) + 1]
                I_y = _grad_y[o_y + _yi][o_x - math.floor(k_w / 2): o_x + math.floor(k_w / 2) + 1]
                I_t = _grad_t[o_y + _yi][o_x - math.floor(k_w / 2): o_x + math.floor(k_w / 2) + 1]
                I_x_k[_yi] = I_x
                I_y_k[_yi] = I_y
                I_t_k[_yi] = I_t

            I_x_k_sum = signal.convolve2d(I_x_k, kernel_sum, boundary='symm', mode='same').flatten()
            I_y_k_sum = signal.convolve2d(I_y_k, kernel_sum, boundary='symm', mode='same').flatten()
            I_t_k_sum = signal.convolve2d(I_t_k, kernel_sum, boundary='symm', mode='same').flatten()

            A = np.asmatrix(np.array([[I_x_k_sum], [I_y_k_sum]])).transpose()
            b = np.array([I_t_k_sum]).transpose()
            A_sqr = A.T * A
            v = np.linalg.pinv(A_sqr) * A.T * b

            # # TODO: Aperture problem condition, commented out since not sure how the eigenvalues are supposed to scale
            # eigenvalue_1 = np.min(abs(np.linalg.eigvals(A_sqr)))
            # if eigenvalue_1 >= 90450873:
            #     continue

            # Normalized then scaled vector for displaying purposes
            v_norm = np.array(v) / (np.sqrt(np.sum(np.array(v) ** 2)) + eps) * 15
            color = (0, 0, np.linalg.norm(v) * 255)

            _orig_frame = cv2.arrowedLine(_orig_frame, (_x, _y), (_x + int(v_norm[0]), _y + int(v_norm[1])), color, 4)

    return _orig_frame


def get_index(_x, _y, _h):
    return _h * _y + _x


width = 480
height = 270
k_size = 13

cap = cv2.VideoCapture('videos/video.mov')
cap.set(3, width)
cap.set(4, height)

ret, frame = cap.read()
last_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

n = 7
x = np.linspace(0, width - 1, n, dtype=int)
y = np.linspace(0, height - 1, n, dtype=int)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

out = cv2.VideoWriter('videos/output001.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        if i % 2 == 0:
            i += 1
            continue

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        step_frame = np.copy(frame)

        frame = lucas_kanade(frame, last_frame, x, y, n)
        last_frame = frame

        cv2.imshow('Frame', frame)
        out.write(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

    i += 1

cap.release()
out.release()
cv2.destroyAllWindows()
