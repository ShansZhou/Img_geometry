import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# detect corners from chessboard
def detectCorners(cb_folder_path, CB_width=9, CB_height=6):

    corners_list = []

    imgpath_list = os.listdir(cb_folder_path)

    for imgpath in imgpath_list:

        img = cv2.imread("./data/chessboards/"+imgpath, cv2.IMREAD_GRAYSCALE)

        ret, corners = cv2.findChessboardCorners(img, (CB_width, CB_height), None)
        
        corners_xy = np.squeeze(corners, axis=1)
        corners_list.append(corners_xy)


    return corners_list

# generate world coordinate wrt chessboard
def genCBpoints(CB_width=9, CB_height=6, interval_mm = 10.0):
    CB_points = []
    for h in range(CB_height):
        for w in range(CB_width):
            x = w*interval_mm
            y = h*interval_mm
            CB_points.append([x,y])
    
    return CB_points

# calculate homography matrix
def calHomographyMatrix(CBpoints, detctPoints_list):

    homography_list = []
    for detctPts in detctPoints_list:
        A = []
        for idx in range(len(detctPts)):
            x_cb = CBpoints[idx][0]
            y_cb = CBpoints[idx][1]
            x = detctPts[idx][0]
            y = detctPts[idx][1]

            # Matrix: 2x9
            A.append([
                        [x_cb, y_cb,  1,    0,    0,    0,   -x_cb*x, - y_cb*y, x],
                        [   0,    0,  0, x_cb, y_cb,    1,   -x_cb*y,  -y_cb*y, y]
                    ])
        
        # SVD
        U, S, V = np.linalg.svd(np.reshape(A,(len(detctPts)*2,9)))

        curr_H = np.reshape(V[-1], (3,3))

        homography_list.append(curr_H)

    return homography_list

# calculate intrinsic matrix
def v(i, j, H):
    return np.array([
        H[i,0]*H[j,0],
        H[i,0]*H[j,1]+H[i,1]*H[j,0],
        H[i,0]*H[j,2]+H[i,2]*H[j,0],
        H[i,1]*H[j,1],
        H[i,1]*H[j,2]+H[i,2]*H[j,1],
        H[i,2]*H[j,2]
        ])

def calIntrinsicMatrix(homography_list):
    h_count = len(homography_list)

    vec = []
    
    # apply each homography to the equation
    for homos in homography_list:
        vec.append(v(0,1,homos))
        vec.append(v(0,0,homos) - v(1,1,homos))

    # solve least square for each homography, and find b vector
    b = np.linalg.lstsq(np.array(vec), np.zeros(h_count*2))[-1]

    # B: 3x3, diagonal matrix, only 6 variables are needed
    B = np.array([[b[0],b[1],b[2]],
                  [b[1],b[3],b[4]],
                  [b[2],b[4],b[5]]])
    
    # calcuate intrinsic parameters
    d = B[0,0]*B[1,1]-B[0,1]**2
    d = -d

    fx = np.sqrt(1/B[0,0])
    fy = np.sqrt(B[0,0]/d)
    cx = (B[0,2]*B[0,1] - B[0,0]*B[1,2]) / d
    gamma = -B[0,1]*(fx**2)*fy
    cy = ((gamma*cx)/fy) - (B[0,2]*fx**2)

    # assign intrinsic matrix
    return np.array([
        [ fx,  gamma,  cx],
        [0.0,     fy,  cy],
        [0.0,    0.0, 1.0]
        ])

# calcuate each extrinsic matrix (every chessboard has a unique extrinsic matrix) based on intrinsic matrix
def calExtrinsicMatrix(intrinsicMatrix, homography_list):
    
    extrinsic_mats = []
    inv_intrinsicMat = np.linalg.inv(intrinsicMatrix)
    for homos in homography_list:
        h0 = homos[:,0]
        h1 = homos[:,1]
        h2 = homos[:,2]

        ld0 = np.linalg.norm(np.dot(inv_intrinsicMat,h0))
        ld1 = np.linalg.norm(np.dot(inv_intrinsicMat,h1))
        ld2 = (ld0+ld1)/2

        r0 = ld0 * np.dot(inv_intrinsicMat, h0) 
        r1 = ld1 * np.dot(inv_intrinsicMat, h1) 
        r2 = np.cross(r0,r1) # 2D coordinate
        t  = ld2 * np.dot(inv_intrinsicMat, h2)

        RT = np.array([
                       np.transpose(r0),
                       np.transpose(r1),
                       np.transpose(r2),
                       np.transpose(t)
                      ])
        RT = np.transpose(RT)

        extrinsic_mats.append(RT)

    return extrinsic_mats

# estimate lens distortion
def estLensDistortion(intrinsic, extrinsic, CBpoints, detctPoints_list):
    uc = intrinsic[0,2]
    vc = intrinsic[1,2]
    D = []
    d = []

    for i in range(0, len(detctPoints_list)):
        for j in range(0, len(CBpoints)):
            cb_pt = np.array([CBpoints[j][0], CBpoints[j][1], 0, 1])
            homo_coords = np.dot(extrinsic[i], cb_pt)
            coords = homo_coords/ homo_coords[-1] # z-axis normalize
            r = np.sqrt(coords[0]**2 + coords[1]**2)
            pts_proj = np.dot(intrinsic, homo_coords)
            pts_proj = pts_proj/pts_proj[2]
            [u, v, _] = pts_proj

            du = u -uc
            dv = v -vc

            # apply distortion equation
            D.append(np.array([du*r**2, du*r**4]))
            D.append(np.array([dv*r**2, dv*r**4]))
            
            u_pixel = detctPoints_list[i][j][0]
            v_pixel = detctPoints_list[i][j][1]

            d.append(u_pixel-u)
            d.append(v_pixel-v)
    
    k = np.linalg.lstsq(
        np.array(D),
        np.array(d)
    )

    return k[-1]


################### test part ###################

corners_list = detectCorners("data\chessboards", CB_width=9, CB_height=6)
CB_points = genCBpoints(CB_width=9, CB_height=6, interval_mm=10)
homographies = calHomographyMatrix(CBpoints=CB_points, detctPoints_list=corners_list)
intrinsic_mat = calIntrinsicMatrix(homographies)
extrinsicMat_list = calExtrinsicMatrix(intrinsicMatrix=intrinsic_mat, homography_list=homographies)
distortions = estLensDistortion(intrinsic_mat,extrinsicMat_list, CB_points, corners_list)
print(distortions)