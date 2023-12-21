import numpy as np
import AffineTransforms

def trasnlateImg(src_img, vec=[0,0]):
    return AffineTransforms.translation(src_img, vec)

def scalingImg(src_img, scalar=[1,1]):
    return AffineTransforms.scaling(src_img, scalar)

def rotatingImg(src_img, degree = 0.0):
    cos_d = np.cos(degree/180.0*np.pi)
    sin_d = np.sin(degree/180.0*np.pi)

    H, W = np.shape(src_img)
    # translate src to center

    # rotate img wrt center
    matrix = np.array([[ cos_d,  sin_d, (1-cos_d)*(0.5*W) - sin_d*(0.5*H)],
                       [-sin_d,  cos_d, sin_d*(0.5*W) + (1-cos_d)*(0.5*H)],
                       [    0,      0,                                 1]])

    # translate back to src
    return AffineTransforms.affineTransforming(src_img, AffineMatrix=matrix)


