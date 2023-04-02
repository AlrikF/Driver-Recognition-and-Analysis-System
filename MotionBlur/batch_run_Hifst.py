# Run a folder of  image and display the blurmap as the output with input folder and output path as the argparse parameters
import cv2
import HIFST_utilities
import helper_utils
from  scipy import fftpack
import os
import argparse
from matplotlib import pyplot
from pathlib import Path

ap = argparse.ArgumentParser()
# Add the arguments to the parser
ap.add_argument("-i","--input_file_path", required= True, help="path to image")
ap.add_argument("-op","--output_file_path",required= True, help="Path to output")
ap.add_argument("-s","--slide_width",required= True, help="Shift in pixels while using different kernels")
ap.add_argument("-ss",'--sig_s', nargs='?', const=15, type=int)
ap.add_argument("-sr",'--sig_r', nargs='?', const=0.25, type=float)
ap.add_argument("-ri",'--recursive_iterations', nargs='?', const=3, type=int)
ap.add_argument("-od",'--odd_kernel', nargs='?', const=1, type=int)
args = vars(ap.parse_args())



# Give the name of the folder in which u want to
# file_path="D:/BE_PROJECT/image/  "
# op_path ="D:/BE_PROJECT/image_op/"

file_path = args["input_file_path"]

Path(     os.path.join(args["output_file_path"],str(args["slide_width"])+"_ss_"+str(args["sig_s"])+"_sr_"+str(args["sig_r"])+"_ri_"+str(args["recursive_iterations"])+"_o_"+str(args["odd_kernel"]))).mkdir(parents=True, exist_ok=True)
op_path = os.path.join(args["output_file_path"],str(args["slide_width"])+"_ss_"+str(args["sig_s"])+"_sr_"+str(args["sig_r"])+"_ri_"+str(args["recursive_iterations"])+"_o_"+str(args["odd_kernel"]))

# op_path   = args["output_file_path"]

h_obj =helper_utils.helper_utils()
hfu_obj= HIFST_utilities.HIFST_utilities()

print(op_path)

for file in os.listdir(file_path):
    print(file_path+file)
    im = (cv2.imread(file_path+file,0))
    # im = h_obj.square_and_center(im)
    cv2.imwrite(os.path.join(op_path ,"{}_ori.jpg".format(file[:-4])), im)
    try:
        FinalMap = hfu_obj.Hifst(im,args["slide_width"],args["sig_s"],args["sig_r"],args["recursive_iterations"],odd_Flag=args["odd_kernel"])
        print(os.path.join(op_path,"{}_op.jpg".format(file[:-4])))

        pyplot.imsave(os.path.join(op_path,"{}_op.jpg".format(file[:-4])),FinalMap)
        cv2.imwrite(os.path.join(op_path,"{}_cv_op.jpg".format(file[:-4])), FinalMap*255)
    except:
        continue

