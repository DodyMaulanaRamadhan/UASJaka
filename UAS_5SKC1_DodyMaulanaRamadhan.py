import cv2
import os

# Buat direktori penyimpanan jika belum ada
output_dir = "captured_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ambil gambar dari webcam
ret, frame = cap.read()
if not ret:
    print("Gagal mengambil gambar.")
else:
    # Simpan gambar asli (original)
    original_path = os.path.join(output_dir, "original.jpg")
    cv2.imwrite(original_path, frame)

    # Deteksi wajah dan crop gambar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Ambil wajah pertama yang terdeteksi dan lakukan cropping
        (x, y, w, h) = faces[0]
        cropped_face = frame[y:y + h, x:x + w]
        cropped_path = os.path.join(output_dir, "cropped_original.jpg")
        cv2.imwrite(cropped_path, cropped_face)
    else:
        print("Tidak ada wajah terdeteksi.")

    # Simpan gambar grayscale
    grayscale_path = os.path.join(output_dir, "grayscale.jpg")
    cv2.imwrite(grayscale_path, gray)

    # Simpan gambar black & white (binary)
    _, blackwhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    blackwhite_path = os.path.join(output_dir, "blackwhite.jpg")
    cv2.imwrite(blackwhite_path, blackwhite)

    print("Gambar berhasil disimpan.")

# Lepas kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()