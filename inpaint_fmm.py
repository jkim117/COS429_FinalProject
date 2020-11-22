import cv2 
import numpy as np
import copy 
import heapq

BAND = 0
KNOWN = 1
INSIDE = 2

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

# point data structure:
'''{
	T:, distance from band border
	I, value for this pixel
	x,
	y,
}'''
class Point():
	def __init__(self, T, I, f, x, y):
		self.T = T # distance from band border
		self.I = I # pixel value
		self.f = f # 0 for BAND, 1 for KNOWN, 2 for INSIDE
		self.x = x
		self.y = y

	def __lt__(self, other):
		return self.T < other.T

def checkBounds(x, y, deltax, deltay, shape):
	if ((x + deltax) < 0 or (x + deltax) >= shape[0] or (y + deltay) < 0 or (y + deltay) >= shape[1]):
		return True # falls outside of bounds
	return False # falls within bounds


def findBand(img_channel, mask):
	boundImg = np.zeros(mask.shape)
	bandHeap = []
	point_image = np.empty(shape = img_channel.shape, dtype = object)

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
				point_image[i, j] = Point(0.0, img_channel[i,j], BAND, i, j)
				boundImg[i,j] = 255
				bandHeap.append(point_image[i, j])
			elif (num_black == 0): # inside boundary
				point_image[i, j] = Point(1.0e6, img_channel[i,j], INSIDE, i, j)
			else: # outside boundary
				point_image[i, j] = Point(0.0, img_channel[i,j], KNOWN, i, j)

	heapq.heapify(bandHeap)
	#cv2.imshow('navier stokes', boundImg)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return bandHeap, point_image
						
def inpaint_fmm_single(point, point_image, img_grad, eps):
	Ia = 0.0
	s = 0.0

	for k in range(-eps, eps + 1):
		for l in range(-eps, eps + 1):
			
			if (k == 0 and l == 0):
				continue
			xk = point.x + k
			yl = point.y + l
			if (checkBounds(xk, yl, 0, 0, point_image.shape)):
				continue

			if (point_image[xk, yl].f == KNOWN):
				if (not checkBounds(xk+1, yl, 0, 0, point_image.shape) and
					not checkBounds(xk-1, yl, 0, 0, point_image.shape) and
					not checkBounds(xk, yl+1, 0, 0, point_image.shape) and
					not checkBounds(xk, yl-1, 0, 0, point_image.shape)):	

					r = np.array([xk - point.x, yl - point.y])
					gradT_X = point_image[xk+1, yl].T - point_image[xk-1, yl].T
					gradT_Y = point_image[xk, yl+1].T - point_image[xk, yl-1].T
					gradT = np.array([gradT_X, gradT_Y]) # use same approx as gradI???
					lenr = np.linalg.norm(r)

					direction = np.sum(r * gradT) / lenr
					
					dst = 1 / (lenr * lenr)
					lev = 1 / (1 + np.absolute(point.T - point_image[xk, yl].T))

					w = np.absolute(direction * dst * lev)
					if (w == 0.0):
						w = 1e-6

					#print(w>0)
					#print(w)
					#print(point_image[xk+1, yl].f ,point_image[xk-1, yl].f,point_image[xk, yl+1].f,point_image[xk, yl-1].f)

					#if (point_image[xk+1, yl].f == KNOWN and point_image[xk-1, yl].f == KNOWN and point_image[xk, yl+1].f == KNOWN and point_image[xk, yl-1].f == KNOWN):
					gradI_X = float(point_image[xk+1, yl].I) - float(point_image[xk-1, yl].I)
					gradI_Y = float(point_image[xk, yl+1].I) - float(point_image[xk, yl-1].I)
					gradI = np.array([gradI_X, gradI_Y])		
					#print('hiii')		

					Ia += w * point_image[xk, yl].I #+ np.absolute(np.sum(gradI * r)))
					s += w
	#print(Ia/s)
	point_image[point.x, point.y].I = Ia / s
	return 0
	#print(Ia)
	#print(s)


def solveT(i1, j1, i2, j2, point_image):
	if (checkBounds(i1, j1, 0, 0, point_image.shape) or checkBounds(i2, j2, 0, 0, point_image.shape)):
		return np.inf

	sol = 1.0e6
	if (point_image[i1,j1].f == KNOWN):
		if (point_image[i2,j2] == KNOWN):
			r = np.sqrt(2 - (point_image[i1,j1].T - point_image[i2,j2].T) * (point_image[i1,j1].T - point_image[i2,j2].T))
			s = (point_image[i1,j1].T + point_image[i2,j2].T - r) / 2.0

			if (s >= point_image[i1,j1].T and s >= point_image[i2,j2].T):
				sol = s
			else:
				s += r
				if (s >= point_image[i1,j1].T and s >= point_image[i2,j2]):
					sol = s
		else:
			sol = 1 + point_image[i1,j1].T
		
	elif (point_image[i2,j2].f == KNOWN):
		sol = 1 + point_image[i2,j2].T
	
	return sol


def inpaint_fmm(img, mask, eps = 10):
	new_img = np.zeros(img.shape, dtype=np.uint8)	
	for i in range(img.shape[2]):
		img_channel = img[:,:,i]
		img_grad = cv2.Laplacian(img_channel, cv2.CV_64F)

		bandHeap, point_image = findBand(img_channel, mask)
		
		while(len(bandHeap) > 0):
			bandPoint = heapq.heappop(bandHeap)
			#print(bandPoint.x,bandPoint.y, bandPoint.T)
			point_image[bandPoint.x, bandPoint.y].f = KNOWN

			for k in [-1, 1]:
				for l in [-1, 1]:
					xk = bandPoint.x + k
					yl = bandPoint.y + l

					if (checkBounds(bandPoint.x, bandPoint.y, k, l, mask.shape)):
						continue
					if (point_image[xk, yl].f != KNOWN): # if point is not known
						if (point_image[xk, yl].f == INSIDE): # if point is inside
							point_image[xk, yl].f = BAND

							err = inpaint_fmm_single(point_image[xk, yl], point_image, img_grad, eps) # TODO
							if err == -1:
								point_image[bandPoint.x, bandPoint.y].f = BAND
								point_image[bandPoint.x, bandPoint.y].T += 1
								heapq.heappush(bandHeap, point_image[bandPoint.x, bandPoint.y])
								continue

						# if out of bounds, solveT = INF
						point_image[xk, yl].T = min([solveT(xk - 1, yl, xk, yl - 1, point_image),
													 solveT(xk + 1, yl, xk, yl - 1, point_image),
													 solveT(xk - 1, yl, xk, yl + 1, point_image),
													 solveT(xk + 1, yl, xk, yl + 1, point_image)])

						heapq.heappush(bandHeap, point_image[xk, yl])

		new_channel = np.zeros((img.shape[0], img.shape[1]))
		for k in range(0, img.shape[0]):
			for l in range(0, img.shape[1]):
				new_channel[k,l] = point_image[k,l].I
		
		new_img[:,:,i] = new_channel

	return new_img


mask = create_mask(img, list(dogs.values()), 5)
# mask = create_mask(img, list(person.values()), 10)
print(img.shape)
final_img = inpaint_fmm(img, mask, eps = 3)

fast_marching = cv2.inpaint(img, mask, 2, cv2.INPAINT_TELEA)
#navier_stokes = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)

# cv2.imshow('fast marching', fast_marching)
cv2.imshow('navier stokes', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

