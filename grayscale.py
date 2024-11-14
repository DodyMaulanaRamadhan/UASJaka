import cv2
import os
from datetime import datetime

img = cv2.imread('KopiTanjung.jpeg')

cv2.imshow("Original Image", img)
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", grayscale)
(th, bw) = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("BW Image", bw)

# Folder penyimpanan
storage_folder = "storage"
os.makedirs(storage_folder, exist_ok=True)

# Dapatkan timestamp saat ini
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

#Simpan gambar Grayscale dengan timestamp
grayscale_filename = os.path.join(storage_folder, f"grayscale_{timestamp}.jpg")
cv2.imwrite(grayscale_filename, grayscale)

# Simpan gambar BW dengan timestamp
bw_filename = os.path.join(storage_folder, f"bw_{timestamp}.jpg")
cv2.imwrite(bw_filename, bw)

cv2.waitKey(0)
cv2.destroyAllWindows()