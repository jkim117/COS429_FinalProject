import cv2 

# img = cv2.imread('test.jpg')
img = cv2.imread('dog.jpg')

# get mask from rCNN
mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)

# fast_marching = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
navier_stokes = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

# cv2.imshow('fast marching', fast_marching)
cv2.imshow('navier stokes', navier_stokes)
cv2.waitKey(0)
cv2.destroyAllWindows()