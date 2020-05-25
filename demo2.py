import face_recognition
known_image = face_recognition.load_image_file("./photos/TomHanks.jpg")
unknown_image = face_recognition.load_image_file("./photos/TomHanks2.jpg")

known_encoding = face_recognition.face_encodings(known_image)
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces(known_encoding, unknown_encoding)

distance = face_recognition.face_distance(known_encoding, unknown_encoding)

print("Result: {}".format(results))
print("Distance: {}".format(distance))
