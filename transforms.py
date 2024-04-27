import random
import cv2
import numpy as np
import math
import torch
import torch.nn.functional as F


class DualCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class NoTransform(object):
    def __call__(self, *args):
        return args[0]


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, img, mask=None):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std

        import cv2
        # cv2.imwrite('./sessions/3.png', img[0, :, :] * 100)
        return img


class OneOf(object):
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img):
        if self.state['p'] < self.prob:
            img = self.state['t'](img)
        return img

    def randomize(self):
        self.state['p'] = random.random()
        self.state['t'] = random.choice(self.transforms)
        self.state['t'].prob = 1.


class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            self.output_size = output_size
        else:
            raise ValueError('Incorrect value')
        # self.keep_size = keep_size
        # self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img):
        rows_in, cols_in = img.shape[1:]
        rows_out, cols_out = self.output_size
        rows_out = min(rows_in, rows_out)
        cols_out = min(cols_in, cols_out)

        r0 = math.floor(self.state['r0f'] * (rows_in - rows_out))
        c0 = math.floor(self.state['c0f'] * (cols_in - cols_out))
        r1 = r0 + rows_out
        c1 = c0 + cols_out

        img = np.ascontiguousarray(img[:, r0:r1, c0:c1])
        # cv2.imwrite('./sessions/5.png', img[0, :, :])
        return img

    def randomize(self):
        # self.state['p'] = random.random()
        self.state['r0f'] = random.random()
        self.state['c0f'] = random.random()


class Scale(object):
    def __init__(self, ratio_range=(0.7, 1.2), prob=.5):
        self.ratio_range = ratio_range
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img):
        """
        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            ch, d0_i, d1_i = img.shape
            d0_o = math.floor(d0_i * self.state['r'])
            d0_o = d0_o + d0_o % 2
            d1_o = math.floor(d1_i * self.state['r'])
            d1_o = d1_o + d1_o % 2

            # img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit,
            #                           borderType=cv2.BORDER_REFLECT_101)
            # img = np.squeeze(img)

            img0 = cv2.resize(img[0, :, :], (d1_o, d0_o))
            img1 = cv2.resize(img[1, :, :], (d1_o, d0_o))
            img2 = cv2.resize(img[2, :, :], (d1_o, d0_o))
            img_final = np.empty((3, *(d0_o, d1_o)), dtype=img.dtype)
            img_final[0, :, :] = img0
            img_final[1, :, :] = img1
            img_final[2, :, :] = img2


            # img = cv2.resize(img, (d1_o, d0_o), interpolation=cv2.INTER_LINEAR)
            # img = img[None, ...]
            # print('end', img_final.shape)
            # cv2.imwrite('./sessions/4.png', img[0, :, :])
        return img_final

    def randomize(self):
        self.state['p'] = random.random()
        self.state['r'] = round(random.uniform(*self.ratio_range), 2)


class Rotate(object):
    def __init__(self, degree_range=(-30., 30.), prob=0.5):
        self.theta_range = torch.deg2rad(torch.Tensor(degree_range))
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, image):
        """
        Parameters
        ----------
        image: (CH, R, C, S) 4D Tensor
        """

        if self.state["p"] < self.prob:
            center_pt_x = int((image.shape[1]) / 2)
            center_pt_y = int((image.shape[2]) / 2)
            center_pt = np.array([center_pt_x, center_pt_y])

            M = cv2.getRotationMatrix2D((center_pt_x, center_pt_y), self.state["theta"].item(), scale=1)
            M[0, 2] = center_pt[0] / image.shape[2]
            M[1, 2] = center_pt[1] / image.shape[1]

            (h, w) = (image.shape[1], image.shape[2])

            ret_tmp = image.copy()
            ret0 = ret_tmp[0, :, :]
            ret1 = ret_tmp[1, :, :]
            ret2 = ret_tmp[2, :, :]

            ret0 = cv2.warpAffine(ret0, M, (w, h))
            ret1 = cv2.warpAffine(ret1, M, (w, h))
            ret2 = cv2.warpAffine(ret2, M, (w, h))

            ret_final = np.empty((3, *(h, w)), dtype=ret_tmp.dtype)
            ret_final[0, :, :] = ret0
            ret_final[1, :, :] = ret1
            ret_final[2, :, :] = ret2

            return ret_final

        else:
            return image

    def randomize(self):
        self.state["p"] = random.random()
        self.state["theta"] = random.uniform(*self.theta_range)


class HorizontalFlip(object):
    def __init__(self, prob=.5):
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img):
        """
        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            img = np.flip(img, axis=1)
        return img

    def randomize(self):
        self.state['p'] = random.random()

