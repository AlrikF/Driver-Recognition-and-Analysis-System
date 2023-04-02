

#All universal basic functions for images

class helper_utils:
    def __init__(self):
        pass

    def center_luminance(self,im):
        im = im.astype(int)
        im = im - 128
        return im




    def square_and_center(self,im):
        side = min(im.shape[0], im.shape[1]) // 2
        im = im[im.shape[0] // 2 - side:im.shape[0] // 2 + side, im.shape[1] // 2 - side:im.shape[1] // 2 + side]
        return im