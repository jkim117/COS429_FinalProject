import cv2 
import numpy as np
import copy 
import heapq

BAND = 0
KNOWN = 1
INSIDE = 2

img = cv2.imread('test.jpg')

# person = np.load('persons.npz')

dogs = np.load('dogs.npz')

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
	# i =0

	for mask in mask_list:
		# if i ==0:
			comb_mask[mask] = 255
			# i += 1

	comb_mask = pad_mask(shape, comb_mask, pad_size)

	return comb_mask

class Point():
	def __init__(self, conf, data, I, f, x, y):
		self.conf = conf # C(p)
		self.data = data # D(p)
		self.I = I # pixel value
		self.f = f # 0 for BAND, 1 for KNOWN, 2 for INSIDE
		self.x = x
		self.y = y

	def __lt__(self, other):
		return (self.conf * self.data) > (other.conf * other.data)

def checkBounds(x, y, deltax, deltay, shape):
	if ((x + deltax) < 0 or (x + deltax) >= shape[0] or (y + deltay) < 0 or (y + deltay) >= shape[1]):
		return True # falls outside of bounds
	return False # falls within bounds

def findBand(img, mask, first=False, point_image=None):
	print(np.sum(mask/255.0))
	bandHeap = []
	if first:
		point_image = np.empty(shape = img.shape[:2], dtype = object)

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):

			num_black = 0
			num_white = 0
			for k in [-1, 1]:
				for l in [-1, 1]:
					if (checkBounds(i, j, k, l, mask.shape)):
						num_black += 1
						continue
					
					if (mask[i + k, j + l] == 255):
						num_white += 1
					else:
						num_black += 1

			if (num_black > 0 and num_white > 0): # boundary
				if first:
					point_image[i, j] = Point(0.0, 0.0, img[i,j,:], BAND, i, j)
				bandHeap.append(point_image[i, j])
			elif (num_black == 0 and first): # inside boundary
				point_image[i, j] = Point(0.0, 0.0, img[i,j,:], INSIDE, i, j)
			elif first: # outside boundary
				point_image[i, j] = Point(1.0, 0.0, img[i,j,:], KNOWN, i, j)

	return bandHeap, point_image

# min dist between bandPoint neighborhood and neighborhood within known
def find_min_neighborhood(x, y, patch, known_patch_mask, point_img, eps):
	min_dist = np.inf
	min_middle = (-1, -1) 

	# for i in range(eps, point_img.shape[0] - eps):
	# 	for j in range(eps, point_img.shape[1] - eps):
	for i in range(x-100, x+101):
		for j in range(y-100, y+101):

			break_iter = False


			comp_patch = np.zeros((2*eps+1, 2*eps+1, 3))

			for deltax in range(-eps, eps+1):
				if break_iter:
					break

				for deltay in range(-eps, eps+1):
					if point_img[i+deltax, j+deltay].f != KNOWN:
						break_iter = True
						break

					comp_patch[deltax+eps, deltay+eps, :] = point_img[i+deltax, j+deltay].I


			if (np.sum(comp_patch) == 0):
				continue 

			curr_dist = np.sum(((patch - comp_patch)**2) * known_patch_mask)
					 
			if (min_dist > curr_dist):
				min_dist = curr_dist
				min_middle = (i, j)

	return min_middle

def inpaint_exemplar(img, mask, eps = 9):
	new_img = np.zeros(img.shape, dtype=np.uint8)
	bandHeap, point_image = findBand(img, mask, True)

	area = (eps*2+1)**2

	while(len(bandHeap) > 0):
		print(len(bandHeap))
		# compute data terms
		for point in bandHeap:
			x = point.x
			y = point.y
			if (checkBounds(x, y, 1, 1, img.shape)):
				continue

			# find normal 
			grad_X = cv2.Scharr(mask, cv2.CV_64F, 1, 0)
			grad_X = cv2.convertScaleAbs(grad_X).astype(float)

			grad_Y = cv2.Scharr(mask, cv2.CV_64F, 0, 1)
			grad_Y = cv2.convertScaleAbs(grad_Y).astype(float)
			n = np.array([grad_X[x, y], grad_Y[x, y]])
			n /= np.linalg.norm(n)
			n = np.reshape(n, (len(n), 1))

			# find image gradient
			gradI_X = cv2.Scharr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0)
			gradI_X = cv2.convertScaleAbs(gradI_X).astype(float)

			gradI_Y = cv2.Scharr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1)
			gradI_Y = cv2.convertScaleAbs(gradI_Y).astype(float)

			total_conf = 0.0
			max_grad = np.zeros(2)
			max_norm = -1
			for i in range(x-eps, x+eps+1):
				for j in range(y-eps, y+eps+1):
					if not checkBounds(i, j, 0, 0, img.shape):
						total_conf += point_image[i, j].conf

						curr_norm = np.sum(gradI_X[i, j]**2 + gradI_Y[i, j]**2)
						if curr_norm > max_norm:
							max_grad = [i, j]
							max_norm = curr_norm

			point.conf = total_conf / area

			gradI = np.array([max_grad[1], -max_grad[0]])
			point.data = np.abs(np.sum(n * gradI)) / 255	

		heapq.heapify(bandHeap)
		# use the boundary point with the highest priority 
		bandPoint = heapq.heappop(bandHeap)
		x = bandPoint.x
		y = bandPoint.y
		point_image[x, y].f == KNOWN

		patch = np.zeros((2*eps+1, 2*eps+1, 3))
		known_patch_mask = np.zeros((2*eps+1, 2*eps+1, 3))

		for i in range(x-eps, x+eps+1):
			for j in range(y-eps, y+eps+1):
				if not checkBounds(i, j, 0, 0, point_image.shape) and point_image[i, j].f == KNOWN:
					patch[i-x+eps, j-y+eps, :] = point_image[i, j].I
					known_patch_mask[i-x+eps, j-y+eps, :] = [1, 1, 1]

		best_x, best_y = find_min_neighborhood(x, y, patch, known_patch_mask, point_image, eps)

		# inpaint, update img and point_img
		for deltax in range(-eps, eps+1):
			for deltay in range(-eps, eps+1):
				if not checkBounds(x+deltax, y+deltay, 0, 0, point_image.shape) and point_image[x+deltax, y+deltay].f != KNOWN:
					point_image[x+deltax, y+deltay].I = img[best_x+deltax, best_y+deltay, :]
					img[x+deltax, y+deltay, :] = img[best_x+deltax, best_y+deltay, :]
					point_image[x+deltax, y+deltay].f = KNOWN
					point_image[x+deltax, y+deltay].conf = point_image[x, y].conf
					mask[x+deltax, y+deltay, 0] = 0

		bandHeap, point_image = findBand(img, mask, point_image=point_image)
	return new_img

mask = create_mask(img, list(dogs.values()), 1)
# mask = create_mask(img, list(person.values()), 10)
print(img.shape)
final_img = inpaint_exemplar(img, mask, eps = 4)

# fast_marching = cv2.inpaint(img, mask, 2, cv2.INPAINT_TELEA)
#navier_stokes = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)

# cv2.imshow('fast marching', fast_marching)
# cv2.imshow('navier stokes', final_img)
cv2.imwrite('./DONE.jpg', final_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

