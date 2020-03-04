import cv2
import numpy as np
import math


def image_infos(file_name):
    image = cv2.imread("test/" + str(file_name) + ".jpg")
    red = [0]*256
    green = [0]*256
    blue = [0]*256
    lbpHistogram = [0]*256
    [rows, cols, bit] = image.shape
    size = rows * cols
    grayImg = np.zeros(shape=(rows,cols), dtype=np.uint8)

    #resmin b-g-r histogramı oluşturulurken aynı zamanda grayscaleye dönüştürülüyor.
    for i in range(rows):
        for j in range(cols):
            red[image[i][j][0]] += 1
            green[image[i][j][1]] += 1
            blue[image[i][j][2]] += 1
            grayImg[i][j] = int(image[i][j][0]*0.114 + image[i][j][1]*0.587 + image[i][j][2]*0.299)
    #resmin histogramı resmin boyutuna bölünerek (pixellerin bulunma olasılıkları eldesi) normalizasyon yapılıyor.
    for i in range(256):
        red[i] = red[i] / size
        green[i] = green[i] / size
        blue[i] = blue[i] / size
    #resmin lbp histogramı oluşturuluyor
    allSum = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            summ = 0
            if grayImg[i][j] > grayImg[i-1][j-1]:
                summ += 128
            if grayImg[i][j] > grayImg[i-1][j]:
                summ += 64
            if grayImg[i][j] > grayImg[i-1][j+1]:
                summ += 32
            if grayImg[i][j] > grayImg[i][j-1]:
                summ += 16
            if grayImg[i][j] > grayImg[i][j+1]:
                summ += 8
            if grayImg[i][j] > grayImg[i+1][j-1]:
                summ += 4
            if grayImg[i][j] > grayImg[i+1][j]:
                summ += 2
            if grayImg[i][j] > grayImg[i+1][j+1]:
                summ += 1
            lbpHistogram[summ] += 1
        size2 = (rows-2)*(cols-2)
    for i in range(256):
        lbpHistogram[i] /= size2
    infos = [blue, green, red, lbpHistogram]
    return infos


def create_data(num_of_images):
    f = open("data.txt", "a")
    for k in range(num_of_images):
        x = image_infos("training/" + str(k+1) + ".jpg")
        for i in range(4):
            for j in range(256):
                f.write(str(x[i][j]) + '\n')
    f.close()


def find_similar(img_name):
    img_infos = image_infos(img_name)
    distance = np.zeros(shape=(2, 70), dtype=float)
    with open("data.txt", "r") as f:
        data = f.readlines()
        for k in range(70):
            for i in range(256):
                distance[0][k] += abs(float(data[k*1024 + i]) - img_infos[0][i])
                distance[0][k] += abs(float(data[k*1024 + 256 + i]) - img_infos[1][i])
                distance[0][k] += abs(float(data[k*1024 + 512 + i]) - img_infos[2][i])
                distance[1][k] += abs(float(data[k*1024 + 768 + i]) - img_infos[3][i])

    sorted1 = sorted(distance[0])
    sorted2 = sorted(distance[1])
    show1 = ['a'] * 5
    show2 = ['a'] * 5
    rate1 = [0] * 5
    rate2 = [0] * 5
    for i in range(5):
        for j in range(70):
            if sorted1[i] == distance[0][j]:
                show1[i] = str(j+1) + ".jpg"
                rate1[i] = j + 1
                break
    for i in range(5):
        for j in range(70):
            if sorted2[i] == distance[1][j]:
                show2[i] = str(j+1) + ".jpg"
                rate2[i] = j + 1
                break
    print("\nResme en yakinlar : ")
    print("rgb ye gore")
    print(show1)
    print("lbp ye gore")
    print(show2)
    a = math.floor((img_name - 1) / 10)
    counter = [0]*2
    for i in range(5):
        if math.floor((rate1[i]-1) / 10) == a:
            counter[0] += 1
        if math.floor((rate2[i]-1) / 10) == a:
            counter[1] += 1
    counter[0] /= 5
    counter[1] /= 5
    return counter

successRateRgb = 0
successRateLbp = 0
a = [0.0] * 2
for i in range(70):
    a = find_similar(i+1)
    successRateRgb += a[0]
    successRateLbp += a[1]

successRateRgb /= 70
successRateLbp /= 70
print("Rgb Başarı Oranı = " + str(successRateRgb))
print("Lbp Başarı Oranı = " + str(successRateLbp))