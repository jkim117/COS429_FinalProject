import cv2 
import numpy as np
import copy 
import heapq

BAND = 0
KNOWN = 1
INSIDE = 2

curr_dir = './eval_set/'

bear_img = cv2.imread(curr_dir+'bear.jpg')
bear = np.load(curr_dir+'bear.npz')

banana_img = cv2.imread(curr_dir+'banana.jpg')
banana = np.load(curr_dir+'banana.npz')

elephant_img = cv2.imread(curr_dir+'elephant_people.jpg')
elephant = np.load(curr_dir+'elephant_people.npz')

plane_img = cv2.imread(curr_dir+'plane.jpg')
plane = np.load(curr_dir+'plane.npz')
people = np.load(curr_dir+'people.npz')

airplanes_img = cv2.imread(curr_dir+'airplane.jpg')
airplane = np.load(curr_dir+'airplane.npz')

fence_img = cv2.imread(curr_dir+'fence.jpg')
fence = np.load(curr_dir+'fence.npz')

stuff_img = cv2.imread(curr_dir+'stuff.jpg')
stuff = np.load(curr_dir+'stuff.npz')
ppl = np.load(curr_dir+'ppl.npz')

img = cv2.imread(curr_dir+'test.jpg')
person = np.load(curr_dir+'persons.npz')
dogs = np.load(curr_dir+'dogs.npz')

motor_img = cv2.imread(curr_dir+'motor.jpg')
more_ppl = np.load(curr_dir+'more_ppl.npz')
motor = np.load(curr_dir+'motor.npz')

donut_img = cv2.imread(curr_dir+'donut.jpg')
donut = np.load(curr_dir+'donut.npz')
knife = np.load(curr_dir+'knife.npz')

def rgbToHsl(img):
	hsl = np.zeros(img.shape)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			r = float(img[i,j,0]) / 255
			g = float(img[i,j,1]) / 255
			b = float(img[i,j,2]) / 255

			pixelMax = max(r,g,b)
			pixelMin = min(r,g,b)

			l = (pixelMax + pixelMin) / 2.0
			h = 0
			s = 0

			if pixelMax != pixelMin:
				d = pixelMax - pixelMin
				if l > 0.5:
					s = d / (2 - pixelMax - pixelMin)
				else:
					s = d / (pixelMax + pixelMin)

				if pixelMax == r:
					if g < b:
						h = (g - b) / d + 6
					else:
						h = (g - b) / d

				elif pixelMax == g:
					h = (b - r) / d + 2
				else:
					h = (r - g) / d + 4
				
				h /= 6
			hsl[i,j,:] = [h * 255,s * 255,l * 255]
	return hsl.astype(np.uint8)

def hueToRgb(m1, m2, h):
	if h < 0:
		h = h + 1
	elif h > 1:
		h = h - 1
	
	if (h * 6 < 1):
		return m1 + (m2 - m1) * h * 6
	if (h * 2 < 1):
		return m2
	if (h * 3 < 2):
		return m1 + (m2 - m1) * (0.66666 - h) * 6
	return m1

def hslToRgb(img):
	rgb = np.zeros(img.shape)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			h = float(img[i,j,0]) / 255.0
			s = float(img[i,j,1]) / 255.0
			l = float(img[i,j,2]) / 255.0

			if l <= 0.5:
				m2 = l * (s + 1)
			else:
				m2 = l + s - l * s
			m1 = l * 2 - m2

			r = hueToRgb(m1, m2, h + 1/3) * 255
			g = hueToRgb(m1, m2, h) * 255
			b = hueToRgb(m1, m2, h - 1/3) * 255

			rgb[i,j,:] = [r,g,b]

	return rgb.astype(np.uint8)


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
	#i =0

	for mask in mask_list:
		#if i ==0:
		comb_mask[mask] = 255
		# break
			#i += 1

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

def updateBand(bandHeapList, eps, x, y, point_image):
	for deltax in range(-(eps+1), eps+2):
		for deltay in range(-(eps+1), eps+2):
			if not checkBounds(x+deltax, y+deltay, 0, 0, point_image.shape) and point_image[x+deltax, y+deltay].f == INSIDE:
				point_image[x+deltax,y+deltay].f = BAND
				bandHeapList.append(point_image[x+deltax,y+deltay])
	
	return bandHeapList


# min dist between bandPoint neighborhood and neighborhood within known
def find_min_neighborhood(x, y, patch, known_patch_mask, point_img, eps):
	min_dist = np.inf
	min_middle = (-1, -1) 

	for i in range(eps, point_img.shape[0] - eps):
		for j in range(eps, point_img.shape[1] - eps):
	#for i in range(x-100, x+101):
	#	for j in range(y-100, y+101):
	#for i in range(0, point_img.shape[0]):
		#for j in range(0, point_img.shape[1]):

			break_iter = False


			comp_patch = np.zeros((2*eps+1, 2*eps+1, 3))

			for deltax in range(-eps, eps+1):
				if break_iter:
					break

				for deltay in range(-eps, eps+1):
					if checkBounds(i + deltax, j + deltay, 0, 0, point_img.shape) or point_img[i+deltax, j+deltay].f != KNOWN:
						break_iter = True
						break

					comp_patch[deltax+eps, deltay+eps, :] = point_img[i+deltax, j+deltay].I


			if (np.sum(comp_patch) == 0 or break_iter):
				continue 

			curr_dist = np.sum(((patch - comp_patch)**2) * known_patch_mask)
					 
			if (min_dist > curr_dist):
				min_dist = curr_dist
				min_middle = (i, j)

	if min_middle == (-1, -1):
		print('FATAL ERROR. NO COMP PATCH FOUND. CONSIDER SMALLLER EPS SIZE')
	return min_middle

def inpaint_exemplar(img, mask, eps = 9):
	new_img = np.zeros(img.shape, dtype=np.uint8)
	bandHeap, point_image = findBand(img, mask, True)
	bandHeapList = bandHeap.copy()

	area = (eps*2+1)**2

	counter = 0
	oldX = -1
	oldY = -1

	while(len(bandHeapList) > 0):
		print(len(bandHeapList))
		# compute data terms

		'''counter += 1
		print('counter', counter)
		if counter % 1 == 0:
			cv2.imshow('fast marching', img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			band_image = np.zeros(mask.shape)
			for c in bandHeapList:
				band_image[c.x,c.y] = 255
			cv2.imshow('band', band_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()'''

		# find normal 
		grad_X = cv2.Scharr(mask, cv2.CV_64F, 1, 0)
		grad_X = cv2.convertScaleAbs(grad_X).astype(float)
		grad_Y = cv2.Scharr(mask, cv2.CV_64F, 0, 1)
		grad_Y = cv2.convertScaleAbs(grad_Y).astype(float)

		# find image gradient
		gradI_X = cv2.Scharr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0)
		gradI_X = cv2.convertScaleAbs(gradI_X).astype(float)
		gradI_Y = cv2.Scharr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1)
		gradI_Y = cv2.convertScaleAbs(gradI_Y).astype(float)

		for point in bandHeapList:
			x = point.x
			y = point.y
			if oldX != -1 and oldY != -1 and np.sqrt((oldX-x)**2 + (oldY-y)**2) > np.sqrt(2) * 2 * (eps + 1):
				continue

			if (checkBounds(x, y, 1, 1, img.shape)):
				continue
			
			n = np.array([grad_X[x, y], grad_Y[x, y]])
			n /= np.linalg.norm(n)
			n = np.reshape(n, (len(n), 1))

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
					if point_image[x+deltax,y+deltay].f == BAND:
						bandHeapList.remove(point_image[x+deltax, y+deltay])

					point_image[x+deltax, y+deltay].f = KNOWN
					point_image[x+deltax, y+deltay].conf = point_image[x, y].conf
					mask[x+deltax, y+deltay] = 0

		bandHeap = updateBand(bandHeapList, eps, x, y, point_image)
		oldX = x
		oldY = y
		bandHeapList = bandHeap.copy()

		# cv2.imshow('fast marching', mask)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()


	return img


imgs = [bear_img, banana_img, elephant_img, plane_img, plane_img, airplanes_img, fence_img, stuff_img, stuff_img, img, img, motor_img, motor_img, donut_img, donut_img]
npzs = [bear, banana, elephant, plane, people, airplane, fence, stuff, ppl, person, dogs, more_ppl, motor, donut, knife]

for i in range(4, len(imgs)):
	#if i!=3:
	#	continue
	# len(imgs)
	img = imgs[i]
	npz = npzs[i]
	
	if i == 0 or i ==3:
		mask = create_mask(img, list(npz.values()), 20) # pad 20 with bear and plane, pad 5 with others
	else:
		mask = create_mask(img, list(npz.values()), 5) # pad 20 with bear and plane, pad 5 with others
	mask = np.squeeze(mask, axis = 2)

	mask = cv2.resize(mask, (0, 0), fx=0.5,fy=0.5)
	img = cv2.resize(img, (0, 0), fx=0.5,fy=0.5)

	#cv2.imwrite('./outputs/' + str(i) + 'mask.jpg', mask)
	print(i, 'mask created')
	#img_with_mask = np.copy(img)
	#img_with_mask[mask == 255, :] = [255, 255, 255]

	#cv2.imwrite('./exemplar_outputs/' + str(i)+'image_with_mask.jpg', img_with_mask)

	'''cv2.imshow('mask_display', img_with_mask)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imshow('img_display', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()'''

	final_img = rgbToHsl(img)
	final_img = inpaint_exemplar(final_img, mask, eps = 3)
	#final_img = inpaint_fmm(final_img, mask, eps = 3)
	final_img = hslToRgb(final_img)
	cv2.imwrite('./'+ str(i) + 'DONE_hsl.jpg', final_img)

# mask = create_mask(img, list(dogs.values()), 20)

# resize mask and img
#mask = cv2.resize(mask, (100, 150))
#img = cv2.resize(img, (100, 150))

# mask = create_mask(img, list(person.values()), 10)
# print(img.shape)
# print(mask.shape)
# final_img = inpaint_exemplar(img, mask, eps = 10)

# fast_marching = cv2.inpaint(img, mask, 2, cv2.INPAINT_TELEA)
#navier_stokes = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)

# cv2.imshow('fast marching', fast_marching)
# cv2.imshow('navier stokes', final_img)
# cv2.imwrite('./DONE1.jpg', final_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

