import numpy as np

def legendre_2d(x, y, deg_x, deg_y):
    # not working for some reason
    c = np.zeros([deg_y + 1, deg_x + 1])
    c[-1, -1] = 1
    # leg = np.polynomial.legendre.legval2d(x.reshape(1, -1), y.reshape(-1, 1), c)
    leg = np.polynomial.legendre.leggrid2d(x, y, c)
    return leg


def legendre_2d_legacy(x, y, deg_x, deg_y):
    c_x = np.zeros(deg_x + 1)
    c_y = np.zeros(deg_y + 1)
    c_x[-1] = 1
    c_y[-1] = 1
    leg_x = np.polynomial.legendre.legval(x, c_x)
    leg_y = np.polynomial.legendre.legval(y, c_y)
    return np.matmul(leg_y.reshape(-1, 1), leg_x.reshape(1, -1))

def legendre_2d_3(x, y, c):
    # not working for some reason
    leg = np.polynomial.legendre.leggrid2d(x, y, c)
    return leg

def legendre_fitting(Y, max_degree):
    m = Y.shape[0]
    n = Y.shape[1]

    # y, x = np.mgrid[np.linspace(-1, 1, m), np.linspace(-1, 1, n)]
    y = np.linspace(-1, 1, m)
    x = np.linspace(-1, 1, n)
    c = np.zeros([max_degree + 1, max_degree + 1])
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if i + j < max_degree + 1:
                c[i, j] = 1

    X = np.zeros([m * n, np.sum(c).astype(np.int)])
    iterator = 0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] == 1:
                X[:, iterator] = legendre_2d_legacy(x, y, i, j).reshape(m * n, )
                iterator = iterator + 1

    YY = np.reshape(Y, (m * n, 1))

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

    surface = np.reshape(np.dot(X, theta), (m, n))

    c_new = np.zeros_like(c)
    iterator = 0
    for j in range(c.shape[1]):
        for i in range(c.shape[0]):
            if c[i, j] == 1:
                c_new[i, j] = theta[iterator]
                iterator = iterator + 1

    return surface, c_new


def legendre_fitting_nan(Y, max_degree):
    m = Y.shape[0]
    n = Y.shape[1]

    # y, x = np.mgrid[np.linspace(-1, 1, m), np.linspace(-1, 1, n)]
    y = np.linspace(-1, 1, m)
    x = np.linspace(-1, 1, n)
    c = np.zeros([max_degree + 1, max_degree + 1])
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if i + j < max_degree + 1:
                c[i, j] = 1

    X = np.zeros([m * n, np.sum(c).astype(np.int)])
    iterator = 0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] == 1:
                c_temp = np.zeros_like(c)
                c_temp[i, j] = 1
                # X[:, iterator] = legendre_2d_legacy(x, y, c_temp).reshape(m * n, )
                X[:, iterator] = legendre_2d_legacy(x, y, i, j).reshape(m * n, )
                iterator = iterator + 1

    YY = np.reshape(Y, (m * n, 1))

    # Only this part is added to regular Legendre fitting
    nan_indices = np.isnan(YY)
    nan_indices = nan_indices.reshape([nan_indices.shape[0], ])
    X_no_nan = X[~ nan_indices]
    YY_no_nan = YY[~ nan_indices]
    #####################################################

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X_no_nan.transpose(), X_no_nan)), X_no_nan.transpose()),
                   YY_no_nan)  # swapped variables

    surface = np.reshape(np.dot(X, theta), (m, n))  # surface retrieved for the original size

    c_new = np.zeros_like(c)
    iterator = 0
    for j in range(c.shape[1]):
        for i in range(c.shape[0]):
            if c[i, j] == 1:
                c_new[i, j] = theta[iterator]
                iterator = iterator + 1

    return surface, c_new


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage.io import imread
    from skimage.color import rgb2gray

    im = imread('sample.jpg')
    # im = np.rot90(im)
    im = rgb2gray(im)

    correction, coeffs = legendre_fitting_nan(im, 4)
    plt.rcParams['image.cmap'] = 'gray'
    plt.subplot(131)
    plt.imshow(im)
    plt.subplot(132)
    plt.imshow(correction)
    plt.subplot(133)
    plt.imshow(im - correction)
    # plt.show()
    plt.savefig('result2.jpg', dpi=500)
    test = 1