# # -*- coding: utf-8 -*-
# """
# Created on Sun Dec  1 20:57:50 2019

# @author: Shivam
# """
import cv2
import matplotlib.pyplot as plt
import numpy as np
def main():
# # Implemting Camera Capture
#     camera = cv2.VideoCapture(0)
#     backSub = cv2.createBackgroundSubtractorMOG2()
#     while (1):
#         ret, img = camera.read()
#         if ret == True:
#         # Display the resulting frame
#             cv2.imshow('Frame',img)
#         k = cv2.waitKey(33)
#         if k==27:    # Esc key to stop
#             cv2.destroyAllWindows()
#             break
#         if k==ord('s'):
#             cv2.imwrite('input_img.jpg', img) 
#     del(camera)
# # Reading an image
#     image = cv2.imread('input_img.jpg',1)
# # Converting image into grayscale
#     image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # Applying Median Blur
#     image_grayscale_blurred = cv2.medianBlur(image_grayscale,3)
# # Applying CLAHE
#     clahe = cv2.createCLAHE()
#     image_histogram_equalized = clahe.apply(image_grayscale_blurred)
#     image_histogram_equalized = clahe.apply(image_histogram_equalized)
# # Applying Median Blur on Histogram Equalized Image
#     image_histogram_equalized_blurred = cv2.medianBlur(image_histogram_equalized,5)
# # Applying GABOR Filter for edge detection
#     ksize = 10
#     sigma = 3
#     theta = np.pi / 4
#     lamda = np.pi / 4
#     gamma = 2.3
#     phi = 0.5
#     kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
#     kernel_resized = cv2.resize(kernel, (150, 150))
#     image_gabor_applied = cv2.filter2D(image_histogram_equalized_blurred, cv2.CV_8UC3, kernel)
# # Applying Median Blur on Gabor Filtered Image
#     image_gabor_applied_blurred = cv2.medianBlur(image_gabor_applied, ksize = 13) 
# # Applying OTSU Threshhold
#     ret,OTSU_THRESH_2 = cv2.threshold(image_gabor_applied_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # Applying Median Blur on OTSU image
#     OTSU_THRESH_2 = cv2.medianBlur(OTSU_THRESH_2, ksize = 7)
# # Erosion on OTSU Image
#     eroded_image = cv2.erode(OTSU_THRESH_2, kernel, iterations = 1)
# # Applying Mask on the image
#     output_img = eroded_image.copy()    
#     output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
#     for each_row in range(output_img.shape[0]):
#         for each_col in range(output_img.shape[1]):
#             if (output_img[each_row, each_col][0] == 255 and output_img[each_row, each_col][1] == 255 and  output_img[each_row, each_col][2] == 255):
#                 output_img[each_row, each_col] = np.array([0, 118, 19]) ## Dark lime Green
# # Printing images
#     titles = ['GrayScale', 'Median Filter', 'Histogram Equalization', 'Gabor Filtered', 'OTSU Threshold', 'Erosion']
#     images = [image_grayscale, image_grayscale_blurred, image_histogram_equalized_blurred, image_gabor_applied_blurred, OTSU_THRESH_2, eroded_image]
#     for i in range(len(images)):
#         plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
#     plt.show()



# # Implemeting all the above process in video streaming
# # Video Capturing 
# 	cam_capture = cv2.VideoCapture(0)
# 	# Setting FPS
# 	cam_capture.set(cv2.CAP_PROP_FPS, 60)
# 	cv2.destroyAllWindows()
# 	upper_left = (100, 100)
# 	bottom_right = (450, 350)
# 	while 1:
# 	    _, image_frame = cam_capture.read()	   
# 	    #Rectangle marker
# 	    r = cv2.rectangle(image_frame, upper_left, bottom_right, (100, 50, 200), 5)
# 	    input_image = image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
# 	# Converting image into grayscale
# 	    image_grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# 	# Applying Median Blur
# 	    image_grayscale_blurred = cv2.medianBlur(image_grayscale,3)
# 	# Applying CLAHE
# 	    clahe = cv2.createCLAHE()
# 	    image_histogram_equalized = clahe.apply(image_grayscale_blurred)
# 	    image_histogram_equalized = clahe.apply(image_histogram_equalized)
# 	# Applying Median Blur on Histogram Equalized Image
# 	    image_histogram_equalized_blurred = cv2.medianBlur(image_histogram_equalized,3)
# 	# Applying GABOR Filter
# 	    ksize = 10
# 	    sigma = 3
# 	    theta = np.pi / 4
# 	    lamda = np.pi / 4
# 	    gamma = 2.3
# 	    phi = 0.5
# 	    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
# 	    kernel_resized = cv2.resize(kernel, (150, 150))
# 	    image_gabor_applied = cv2.filter2D(image_histogram_equalized_blurred, cv2.CV_8UC3, kernel)
# 	# Applying Median Blur on Gabor Filtered Image
# 	    image_gabor_applied_blurred = cv2.medianBlur(image_gabor_applied, ksize=13)
# 	# Applying OTSU Threshhold
# 	    ret,OTSU_THRESH_2 = cv2.threshold(image_gabor_applied_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# 	# Erosion on OTSU Image
# 	    eroded_image = cv2.erode(OTSU_THRESH_2, kernel, iterations = 1)
# 	# Applying Mask on the image
# 	    output_img = eroded_image.copy()
# 	    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
# 	    for each_row in range(output_img.shape[0]):
# 	        for each_col in range(output_img.shape[1]):
# 	            if (output_img[each_row, each_col][0] == 255 and output_img[each_row, each_col][1] == 255 and  output_img[each_row, each_col][2] == 255):
# 	                output_img[each_row, each_col] = np.array([0, 118, 19]) ## Dark lime Green

# 	    #Conversion for 3 channels to put back on original image (streaming)
# 	    sketcher_rect_rgb = cv2.cvtColor(image_grayscale, cv2.COLOR_BGR2RGB)	   
# 	    #Replacing the sketched image on Region of Interest
# 	    image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = sketcher_rect_rgb
# 	    cv2.imshow("Vein Detection", image_frame)
# 	    if cv2.waitKey(33) == 27:
# 	        break
# 	cam_capture.release()
# 	cv2.destroyAllWindows()
# if __name__ == "__main__":
#     main()



# Process dealing with same variable
# Implemeting all the above process in video streaming
# Video Capturing 
	cam_capture = cv2.VideoCapture(0)
	# Setting FPS
	cam_capture.set(cv2.CAP_PROP_FPS, 60)
	# cv2.destroyAllWindows()
	upper_left = (100, 100)
	bottom_right = (450, 350)
	while 1:
	    _, image_frame = cam_capture.read()	   
	#Rectangle marker
	    r = cv2.rectangle(image_frame, upper_left, bottom_right, (100, 50, 200), 5)
	    image = image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
	# Converting image into grayscale
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Applying Median Blur
	    image = cv2.medianBlur(image,3)
	# Applying CLAHE
	    clahe = cv2.createCLAHE()
	    image = clahe.apply(image)
	    image = clahe.apply(image)
	# Applying Median Blur on Histogram Equalized Image
	    image = cv2.medianBlur(image,3)
	# Applying GABOR Filter
	    ksize = 10
	    sigma = 3
	    theta = np.pi / 4
	    lamda = np.pi / 4
	    gamma = 2.3
	    phi = 0.5
	    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
	    kernel_resized = cv2.resize(kernel, (150, 150))
	    image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
	# Applying Median Blur on Gabor Filtered Image
	    image = cv2.medianBlur(image, ksize=13)
	# Applying OTSU Threshhold
	    ret,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# Erosion on OTSU Image
	    image = cv2.erode(image, kernel, iterations = 1)
	# Applying Mask on the image
	    output_img = image.copy()
	    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
	    for each_row in range(output_img.shape[0]):
	        for each_col in range(output_img.shape[1]):
	            if (output_img[each_row, each_col][0] == 255 and output_img[each_row, each_col][1] == 255 and  output_img[each_row, each_col][2] == 255):
	                output_img[each_row, each_col] = np.array([0, 118, 19]) ## Dark lime Green

	    #Conversion for 3 channels to put back on original image (streaming)
	    sketcher_rect_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)	   
	    #Replacing the sketched image on Region of Interest
	    image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = sketcher_rect_rgb
	    cv2.imshow("Vein Detection", image_frame)
	    if cv2.waitKey(33) == 27:
	        break
	cam_capture.release()
	cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

