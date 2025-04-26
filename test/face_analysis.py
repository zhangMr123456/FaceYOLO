import time

from insightface.data import get_image

from core.face_analysis import buffalo_model

image_path = r""

b = time.time()
for i in range(100):
    image = get_image(image_path)
    faces = buffalo_model.get(image)
    for face in faces:
        print(face.embedding)

print(time.time() - b)