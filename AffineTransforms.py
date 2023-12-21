import numpy as np


# translation
# img is gray, translation is t = [x, y]
def translation(img, t = [0, 0]):

    tx = t[0]
    ty = t[1]

    # translation matrix
    mat_trans = np.array([[1, 0, tx],
                          [0, 1, ty],
                          [0, 0,  1]])
    
    rows, cols = np.shape(img)
    img_trans = np.zeros((rows, cols), np.uint8)
    for row in range(rows):
        for col in range(cols):
            src_pixel = img[row, col]
            p = np.array([col, row, 1]).transpose()

            # destination: p` = M*p
            p_t = np.dot(mat_trans, p)

            row_new = np.uint16(p_t[1])
            col_new = np.uint16(p_t[0])
            
            # assign to new location
            if col_new<0 or col_new>=cols or row_new<0 or row_new>=rows:
                # img_trans[row, col] =0
                pass
            else:
                img_trans[row_new, col_new] = src_pixel

    return img_trans


# scaling
def scaling(img, s = [0, 0]):

    sx = s[0]
    sy = s[1]

    # scaling matrix
    mat_scale = np.array([[sx,  0, 0],
                          [ 0, sy, 0],
                          [ 0,  0, 1]])
    
    rows, cols = np.shape(img)
    img_scaled = np.zeros((rows, cols), np.uint8)
    for row in range(rows):
        for col in range(cols):
            src_pixel = img[row, col]
            p = np.array([col, row, 1]).transpose()

            # destination: p` = M*p
            p_t = np.dot(mat_scale, p)

            row_new = np.uint16(p_t[1])
            col_new = np.uint16(p_t[0])
            
            # assign to new location
            if col_new<0 or col_new>=cols or row_new<0 or row_new>=rows:continue
            
            img_scaled[row_new, col_new] = src_pixel

    return img_scaled

# affien transforms
# input is gray image, transform matrix (translation, rotation, flip, scaling, etc)
def affineTransforming(img, AffineMatrix):
    rows, cols = np.shape(img)
    img_transformed = np.zeros((rows, cols), np.uint8)
    for row in range(rows):
        for col in range(cols):
            src_pixel = img[row, col]
            p = np.array([col, row, 1.0], np.float32).transpose()
            p_d = np.dot(AffineMatrix, p)
            row_new = np.uint16(p_d[1])
            col_new = np.uint16(p_d[0])
            if row_new<0 or row_new>=rows or col_new<0 or col_new>=cols: continue
            img_transformed[row_new, col_new] = src_pixel
    
    return img_transformed