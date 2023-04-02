# Run a single image and display the blurmap as the output with input image and output path as the argparse parameters

import cv2
import HIFST_utilities
import helper_utils
import DCT_helper
import argparse
import os
from  scipy import fftpack
from matplotlib import pyplot



ap = argparse.ArgumentParser()
# Add the arguments to the parser
ap.add_argument("-i","--image_path", required= True, help="path to image")
ap.add_argument("-op","--output_path",required= True, help="Path to output")
ap.add_argument("-s","--slide_width",required= True, help="Shift in pixels while using different kernels")
ap.add_argument("-ss",'--sig_s', nargs='?', const=15, type=int)
ap.add_argument("-sr",'--sig_r', nargs='?', const=0.25, type=float)
ap.add_argument("-ri",'--recursive_iterations', nargs='?', const=3, type=int)
ap.add_argument("-odd",'--odd_kernel', nargs='?', const=1, type=int)
args = vars(ap.parse_args())

#declare objects
h_obj =helper_utils.helper_utils()
hfu_obj= HIFST_utilities.HIFST_utilities()
dct_obj= DCT_helper.DCT_helper()



im = (cv2.imread(args["image_path"],0))
# im = h_obj.square_and_center(im)
std_deviation_matrix, Mask = dct_obj.calculate_standard_deviation(im)
print(args["odd_kernel"])
FinalMap = hfu_obj.Hifst(im,args["slide_width"],args["sig_s"],args["sig_r"],args["recursive_iterations"],odd_Flag=args["odd_kernel"])

print(os.path.join(args["output_path"],os.path.basename(args["image_path"])).replace("\\","/"))
pyplot.imsave(os.path.join(args["output_path"],os.path.basename(args["image_path"])[:-4]+"op.jpg").replace("\\","/"),FinalMap)
cv2.imwrite(os.path.join(args["output_path"],os.path.basename(args["image_path"])[:-4]+"op_c.jpg").replace("\\","/"),255*FinalMap)

f = pyplot.figure()
pyplot.title("FinalBlurmap")
pyplot.imshow(FinalMap)

cv2.imshow("Final_blurmap",FinalMap)
cv2.waitKey()



