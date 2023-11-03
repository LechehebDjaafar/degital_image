# ----------------Info Developer-------------
# -Last Name : Lecheheb
# -First Name : Djaafar
# -Country : Algeria
# -age : 26
# -Skills : Python - HTML - CSS - C
# -instagram : @ddos_attack_co
# ------------Fallowed Me for instagram-------

#install this
#pip install opencv
#pip install numpy

import cv2
import numpy as np

# 1) قم بتحميل الصورة
IMG = cv2.imread('your_image_path.jpg')

# 2) حدد دقة البكسل للصورة IMG
height, width, _ = IMG.shape

# 3) عرض قيم RGB للبكسل IMG(10, 15)
pixel_value = IMG[10, 15]

# 4) احسب سطوع الصورة IMG
brightness = np.mean(IMG)

# 5) احسب التباين (الكونتراست) للصورة IMG
contrast = np.std(IMG)

# 6) عرض قنوات RGB بشكل منفصل
B, G, R = cv2.split(IMG)

# 7) تحويل IMG إلى مستوى رمادي باستخدام الطريقة المتوسطة
IMG_gray_avg = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)

# 8) تحويل IMG إلى مستوى رمادي باستخدام الصيغة المُرَوَّزَة
# مع حلقات:
IMG_gray_weighted_loop = np.zeros((height, width), dtype=np.uint8)
for i in range(height):
    for j in range(width):
        IMG_gray_weighted_loop[i, j] = 0.299 * IMG[i, j, 2] + 0.587 * IMG[i, j, 1] + 0.114 * IMG[i, j, 0]
# بدون حلقات:
IMG_gray_weighted = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)

# 9) تحديد القيم الدنيا والقصوى لصورتي IMG و IMG2
min_IMG = np.min(IMG)
max_IMG = np.max(IMG)
min_IMG2 = np.min(IMG_gray_weighted)
max_IMG2 = np.max(IMG_gray_weighted)

# 10) حساب دقة البكسل بالميغابكسل
mega_pixels_IMG = (height * width) / 1e6
mega_pixels_IMG2 = (IMG_gray_weighted.size) / 1e6

# 11) دالة لحساب حجم الصورة IMG
def get_image_size(image):
    return image.size

# 12) استخراج وعرض الصورة الفرعية (القص) SUB_IMG من IMG
SUB_IMG = IMG[50:70, 20:90]

# 13) تكييف IMG2 إلى مستويات مختلفة وعرضها
quantized_IMG2_128 = np.uint8(IMG_gray_weighted * 128 / 255)
quantized_IMG2_64 = np.uint8(IMG_gray_weighted * 64 / 255)
quantized_IMG2_32 = np.uint8(IMG_gray_weighted * 32 / 255)

# 14) تحويل الصورة اللونية من نموذج RGB إلى أنماط ألوان مختلفة: HSV، CMYK، و YCbCr
IMG_HSV = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
IMG_CMYK = cv2.cvtColor(IMG, cv2.COLOR_BGR2CMYK)
IMG_YCbCr = cv2.cvtColor(IMG, cv2.COLOR_BGR2YCrCb)

# 15) تحويل IMG2 إلى صور ثنائية باستخدام عتبات مختلفة
binary_IMG2_10 = cv2.threshold(IMG_gray_weighted, 10, 255, cv2.THRESH_BINARY)
binary_IMG2_50 = cv2.threshold(IMG_gray_weighted, 50, 255, cv2.THRESH_BINARY)
binary_IMG2_125 = cv2.threshold(IMG_gray_weighted, 125, 255, cv2.THRESH_BINARY)
binary_IMG2_25 = cv2.threshold(IMG_gray_weighted, 25, 255, cv2.THRESH_BINARY)

# 16) حساب الصورة المكملة لـ IMG2 وإصدارها بتنسيق صورة ثنائية
complementary_IMG2 = 255 - IMG_gray_weighted
complementary_binary_IMG2_10 = 255 - binary_IMG2_10

# 17) إضافة وطرح قيم مختلفة من IMG2
added_IMG2_10 = cv2.add(IMG_gray_weighted, 10)
subtracted_IMG2_10 = cv2.subtract(IMG_gray_weighted, 10)
added_IMG2_30 = cv2.add(IMG_gray_weighted, 30)
subtracted_IMG2_30 = cv2.subtract(IMG_gray_weighted, 30)
added_IMG2_60 = cv2.add(IMG_gray_weighted, 60)
subtracted_IMG2_60 = cv2.subtract(IMG_gray_weighted, 60)
added_IMG2_200 = cv2.add(IMG_gray_weighted, 200)
subtracted_IMG2_200 = cv2.subtract(IMG_gray_weighted, 200)

# 18) زيادة وانخفاض سطوع الصورة بالضرب
increased_brightness_IMG2 = cv2.multiply(IMG_gray_weighted, 1.7)
decreased_brightness_IMG2 = cv2.multiply(IMG_gray_weighted, 0.6)

# 19) تعويض الصورة
inverted_IMG2_2 = cv2.bitwise_not(IMG_gray_weighted)
inverted_IMG2_0= cv2.bitwise_not(IMG_gray_weighted)

# 20) تحميل صورة أخرى (IMG3)
IMG3 = cv2.imread('your_image_path2.jpg')

# 21) حساب القيم الدنيا والقصوى لـ IMG2 و IMG3
min_IMG2 = np.min(IMG_gray_weighted)
max_IMG2 = np.max(IMG_gray_weighted)
min_IMG3 = np.min(IMG3)
max_IMG3 = np.max(IMG3)

# 22) دمج IMG2 و IMG3 بشكل خطي باستخدام أوزان مختلفة
combined_linear_1 = cv2.addWeighted(IMG_gray_weighted, 0.1, IMG3, 0.9, 0)
combined_linear_2 = cv2.addWeighted(IMG_gray_weighted, 0.5, IMG3, 0.5, 0)
combined_linear_3 = cv2.addWeighted(IMG_gray_weighted, 0.9, IMG3, 0.1, 0)

# 23) إنشاء صورتين تحتويان على مربعات
square1 = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(square1, (10, 10), (40, 40), 255, -1)
square2 = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(square2, (30, 30), (60, 60), 255, -1)

# 24) أداء العمليات المنطقية التالية على صور المربعات: AND، OR، XOR، XNOR
logical_and = cv2.bitwise_and(square1, square2)
logical_or = cv2.bitwise_or(square1, square2)
logical_xor = cv2.bitwise_xor(square1, square2)
logical_xnor = cv2.bitwise_not(logical_xor)

# 25) إنشاء صورتين ثنائيتين تحتويان على أشكال تعسفية
arbitrary_shape1 = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(arbitrary_shape1, (50, 50), 30, 255, -1)
arbitrary_shape2 = np.zeros((100, 100), dtype=np.uint8)
cv2.ellipse(arbitrary_shape2, (50, 50), (30, 20), 0, 0, 180, 255, -1)

# 26) أداء عمليات الانحطاط والتوسيع والفتح والإغلاق على الصور الثنائية
kernel = np.ones((3, 3), np.uint8)
erosion1 = cv2.erode(arbitrary_shape1, kernel, iterations=1)
dilation1 = cv2.dilate(arbitrary_shape1, kernel, iterations=1)
opening1 = cv2.morphologyEx(arbitrary_shape1, cv2.MORPH_OPEN, kernel)
closing1 = cv2.morphologyEx(arbitrary_shape1, cv2.MORPH_CLOSE, kernel)
erosion2 = cv2.erode(arbitrary_shape2, kernel, iterations=1)
dilation2 = cv2.dilate(arbitrary_shape2, kernel, iterations=1)
opening2 = cv2.morphologyEx(arbitrary_shape2, cv2.MORPH_OPEN, kernel)
closing2 = cv2.morphologyEx(arbitrary_shape2, cv2.MORPH_CLOSE, kernel)

# 27) إنشاء تأريخ الصورة بالألوان الرمادية لـ IMG2
hist_IMG2 = cv2.calcHist([IMG_gray_weighted], [0], None, [256], [0, 256])

# 28) إضافة 70 إلى جميع بكسلات الصورة ومقارنة تأريخها قبل وبعد العملية
added_70_IMG2 = cv2.add(IMG_gray_weighted, 70)
hist_added_70 = cv2.calcHist([added_70_IMG2], [0], None, [256], [0, 256])

# 29) تحميل صورة منخفضة التباين (IMG4)
IMG4 = cv2.imread('your_image_path3.jpg')

# 30) تطبيق تمدد تأريخ الصورة وعرض التأريخ قبل وبعد كل عملية
hist_IMG4 = cv2.calcHist([IMG4], [0], None, [256], [0, 256])
min_val = np.min(IMG4)
max_val = np.max(IMG4)
stretched_IMG4 = cv2.normalize(IMG4, None, 0, 255, cv2.NORM_MINMAX)
hist_stretched = cv2.calcHist([stretched_IMG4], [0], None, [256], [0, 256])

# 31) تطبيق تصحيح الجاما على IMG2 باستخدام قيم مختلفة للجاما
gamma_1_5 = np.power(IMG_gray_weighted / 255, 1.5) * 255
gamma_4_2 = np.power(IMG_gray_weighted / 255, 4.2) * 255
gamma_2_1 = np.power(IMG_gray_weighted / 255, 2.1) * 255

# 32) تطبيق خوارزمية تعديل تأريخ الصورة على IMG4
equalized_IMG4 = cv2.equalizeHist(IMG4)
hist_equalized = cv2.calcHist([equalized_IMG4], [0], None, [256], [0, 256])

# 33) تحميل صورة أخرى (IMG5)
IMG5 = cv2.imread('your_image_path4.jpg')

# 34) إضافة أنواع مختلفة من الضوضاء إلى IMG5
salt_and_pepper = np.copy(IMG5)
num_salt = np.ceil(0.01 * IMG5.size)
salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in IMG5.shape]
salt_and_pepper[salt_coords] = 1

salt_and_pepper = np.copy(IMG5)
num_pepper = np.ceil(0.01 * IMG5.size)
pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in IMG5.shape]
salt_and_pepper[pepper_coords] = 0

gaussian_noise = np.copy(IMG5)
mean = 0
stddev = 25
noise = np.random.normal(mean, stddev, IMG5.shape).astype(np.uint8)
gaussian_noise = cv2.add(gaussian_noise, noise)

# 35) قم بتطبيق المرشحات المتوسطة والغاوسية على الإصدارات المزعجة من IMG5
mean_filtered = cv2.blur(salt_and_pepper, (5, 5))
gaussian_filtered = cv2.GaussianBlur(gaussian_noise, (5, 5), 0)

# 36) قم بتطبيق المرشحات المتوسطة والغاوسية على الإصدارات المزعجة من IMG5. قم بتطبيق المرشح المتوسط بأحجام مجاورة مختلفة
median_filtered_3 = cv2.medianBlur(salt_and_pepper, 3)
median_filtered_5 = cv2.medianBlur(salt_and_pepper, 5)
median_filtered_7 = cv2.medianBlur(salt_and_pepper, 7)

# 37) تطبيق مشغل Sobel للكشف عن الحافة
sobel_x = cv2.Sobel(IMG5, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(IMG5, cv2.CV_64F, 0, 1, ksize=5)
sobel_edge = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

# عرض النتائج أو حفظها حسب الحاجة
cv2.imshow('IMG', IMG)
# عرض النتائج أو حفظها حسب الحاجة
cv2.waitKey(0)
cv2.destroyAllWindows()
############### Any inquiries, contact me #################################################################
