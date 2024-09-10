import numpy as np
import cv2
from bwm_core import WaterMarkCore

class WatermarkExtractor:
    def __init__(self):
        self.bwm_core = WaterMarkCore()

    def extract_watermark(self, embedded_image_path, watermark_shape):
        # Read the embedded image
        embedded_img = cv2.imread(embedded_image_path, cv2.IMREAD_UNCHANGED)

        if embedded_img is None:
            raise FileNotFoundError(f"Embedded image file '{embedded_image_path}' not found.")

        # Extract the watermark
        self.bwm_core.read_img_arr(img=embedded_img)
        extracted_watermark = self.bwm_core.extract_with_kmeans(img=embedded_img, wm_shape=watermark_shape)

        return extracted_watermark

def main():
    embedded_image_path = 'output_image.png'
    watermark_shape = (100, 150)

    extractor = WatermarkExtractor()
    extracted_watermark = extractor.extract_watermark(embedded_image_path, watermark_shape)

    extracted_watermark_image = (extracted_watermark * 255).astype(np.uint8).reshape(watermark_shape)
    cv2.imwrite('extracted_watermark_program_ver.png', extracted_watermark_image)
    print("Watermark extracted and saved as 'extracted_watermark_program_ver.png'.")

if __name__ == "__main__":
    main()
