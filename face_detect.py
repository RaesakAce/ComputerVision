import cv2 

image_path='698.jpg'

# Read image from your local file system
original_image = cv2.imread(image_path)

# Convert color image to grayscale for Viola-Jones
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Load the classifier and create a cascade object for face detection
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

detected_faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


for (column, row, width, height) in detected_faces:
    cv2.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )

cv2.imshow('Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()