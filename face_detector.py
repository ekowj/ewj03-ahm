import cv2

# Load classifier wajah bawaan OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Baca gambar (ganti "foto.jpg" dengan nama file lo)
img = cv2.imread("PaDodit.jpeg")

# Ubah ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi wajah
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Gambar kotak di sekitar wajah
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Simpan hasil ke file baru
cv2.imwrite("hasil.jpg", img)

print(f"Deteksi wajah selesai! Jumlah wajah terdeteksi: {len(faces)}")
print("Hasil disimpan sebagai 'hasil.jpg'")
