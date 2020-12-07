import sys
import cv2

from inpaint_fmm import inpaint_fmm
from inpaint_exemplar import inpaint_exemplar
from inpaint_hsl import rgbToHsl, inpaint_exemplar, hslToRgb

mask_path = sys.argv[1]
mask = cv2.imread(mask_path)

img_path = sys.argv[2]
img = cv2.imread(img_path)

eps = sys.argv[3]
method = sys.argv[4]
output_path = sys.arg[5]

if method == 'FMM':
	output_img = inpaint_fmm(img, mask, eps)

elif method == 'EXEMPLAR':
	output_img = inpaint_exemplar(img, mask, eps)

elif method == 'HSL':
	output_img = hslToRgb(inpaint_exemplar(rgbToHsl(img), mask, eps))

cv2.imwrite(output_path, output_img)
