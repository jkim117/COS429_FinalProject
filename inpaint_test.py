import cv2 
import numpy as np
import copy 

# img = cv2.imread('test.jpg')
img = cv2.imread('test.jpg')

# get mask from rCNN
# mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
person = np.load('persons.npz')

# person1 = person['x']
# person2 = person['y']

dogs = np.load('dogs.npz')
'''
dog1 = dogs['dog1']
dog2 = dogs['dog2']
dog3 = dogs['dog3']
dog4 = dogs['dog4']


mask = np.zeros((425, 640, 1), dtype=np.uint8) # set everything to black
# mask[person1] = 255
# mask[person2] = 255
mask[dog1] = 255
mask[dog2] = 255
mask[dog3] = 255
mask[dog4] = 255
'''

def pad_mask(shape, mask, pad_size=1):
	height, width, _ = shape
	original_mask = copy.deepcopy(mask)

	for row in range(height):
		for col in range(width): 
			if original_mask[row, col] == 255:
				for row_it in range(-pad_size, pad_size + 1):
					for col_it in range(-pad_size, pad_size + 1):
						check_row = row + row_it
						check_col = col + col_it
						if check_row >= 0 and check_row < height and check_col >= 0 and check_col < width:
							mask[check_row, check_col] = 255

	return mask 

def create_mask(img, mask_list, pad_size=1):
	shape = (*img.shape[:2], 1)
	comb_mask = np.zeros(shape, dtype=np.uint8)

	for mask in mask_list:
		comb_mask[mask] = 255

	comb_mask = pad_mask(shape, comb_mask, pad_size)

	return comb_mask


mask = create_mask(img, list(dogs.values()), 5)
# mask = create_mask(img, list(person.values()), 10)

fast_marching = cv2.inpaint(img, mask, 2, cv2.INPAINT_TELEA)
navier_stokes = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)

# cv2.imshow('fast marching', fast_marching)
cv2.imshow('navier stokes', navier_stokes)
cv2.waitKey(0)
cv2.destroyAllWindows()

