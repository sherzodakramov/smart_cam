from database import Database

db = Database()

db.delete_users()

# from deepface import DeepFace
#
# DeepFace.stream('employees', source=1, enable_face_analysis=False,
#                 detector_backend='retinaface', model_name='VGG-Face')
# # opencv, retinaface, mtcnn, ssd, dlib or mediapipe
