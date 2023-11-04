from database import Database
from redis_db import Memory

db = Database()
red_db = Memory()


db.delete_users()
red_db.open_connection().flushall()

# from deepface import DeepFace
#
# DeepFace.stream('employees', source=1, enable_face_analysis=False,
#                 detector_backend='retinaface', model_name='VGG-Face')
# # opencv, retinaface, mtcnn, ssd, dlib or mediapipe
