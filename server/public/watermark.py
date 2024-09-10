from PIL import Image
from blind_watermark import WaterMark
import numpy as np
import time, sys

from numpy.lib.function_base import extract
import os

def main(arg1):
    watermark_logo = "C:/Users/2h1/Desktop/server/public/watermark.png"
    image_crop(arg1,'C:/Users/2h1/Desktop/server/public/')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00000.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00000.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00001.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00001.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00002.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00002.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00003.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00003.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00004.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00004.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00005.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00005.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00006.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00006.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00007.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00007.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00008.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00008.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00009.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00009.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00010.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00010.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00011.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00011.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00012.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00012.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00013.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00013.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00014.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00014.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00015.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00015.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00016.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00016.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00017.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00017.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00018.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00018.png')
    embed_img = blind_watermark('C:/Users/2h1/Desktop/server/public/00019.png',watermark_logo,'C:/Users/2h1/Desktop/server/public/out/00019.png')
    #embed_img=input("embedded: ")
    #extract_mrk = extract_watermark(embed_img)

    

def blind_watermark(img_orgnl,scram_img,outpath):
    outputPath=outpath
    bwm1 = WaterMark(password_wm=1, password_img=1)
    # read original image
    bwm1.read_img(img_orgnl)
    # read watermark
    bwm1.read_wm(scram_img)
    #embed
    bwm1.embed(outputPath)
    return outputPath

def extract_watermark(embed_img):
    bwm1 = WaterMark(password_wm=1, password_img=1)
    # notice that wm_shape is necessary

    bwm1.extract(filename=embed_img, wm_shape=(30, 50), out_wm_name='3extracted.png', )
    extract_mrk = os.path.basename('C:/Users/2h1/Desktop/server/public/3extracted.png')
    return extract_mrk

def image_crop( infilename, save_path):
    img=Image.open(infilename)
    (img_w,img_h)=img.size
    print(img.size)
    grid_h=1000
    range_h=(int)(img_h/grid_h)
    print(range_h)
    i=0
    h=0
    for h in range(range_h):
        bbox=(0,h*grid_h,1500,(h+1)*(grid_h))
        print(0,h*grid_h,1500,(h+1)*(grid_h))
        crop_img=img.crop(bbox)
        fname="{}.png".format("{0:05d}".format(i))
        savename=save_path+fname
        crop_img.save(savename)
        i+=1

if __name__ == '__main__':
    main(sys.argv[1])
