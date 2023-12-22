import cv2
import matplotlib.pyplot as plt
import numpy as np




## rectify image ##




## computing disparty ##
## TODO: left to right and right to left, correcting bad effects
def estimateDepth_stereo(im_0, im_1,intens_steps=50, startOffset=4, gaus_sigma=1.5, gaus_ksize =11):
    
    # offset and steps
    start = startOffset
    steps = intens_steps # control the accuracy of depth

    # NCC width
    sig = gaus_sigma  # control blur of image
    wid = gaus_ksize   # blur and hole

    img_left = im_0
    img_right = im_1

    # Gaussian filter
    u_left = cv2.mean(img_left)[0]
    u_right = cv2.mean(img_right)[0]

    norm_left = img_left - u_left
    norm_right = img_right - u_right

    # save disparty for each steps
    H, W = np.shape(img_left)
    dmaps = np.zeros((H,W,steps))

    # iterate different disparty in steps
    for displ in range(steps):
        # roll img along x-axis
        norm_left_shifted = np.roll(norm_left, -(displ + start))
        ksize = (wid,wid)
        s = cv2.GaussianBlur(norm_left_shifted*norm_right, ksize, sigmaX=sig, sigmaY=sig)
        s_left = cv2.GaussianBlur(norm_left_shifted*norm_left_shifted, ksize, sigmaX=sig, sigmaY=sig)
        s_right = cv2.GaussianBlur(norm_right*norm_right, ksize, sigmaX=sig, sigmaY=sig)
        dmaps[:,:,displ] = s/np.sqrt(s_left*s_right)

    depth_img = np.argmax(dmaps, axis=2)

    return depth_img

#################################### test part ####################################

# load one frame from each video
img_left = cv2.imread("data\stereo\im2.png", cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread("data\stereo\im6.png", cv2.IMREAD_GRAYSCALE)
depth_img = estimateDepth_stereo(im_0=img_left,im_1=img_right)
fig, _ = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
fig.axes[0].set_title("left")
fig.axes[0].imshow(img_left)
fig.axes[1].set_title("right")
fig.axes[1].imshow(img_right)
fig.axes[2].set_title("depth_img")
fig.axes[2].imshow(depth_img)
plt.show()
print("end")




