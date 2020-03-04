#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

using namespace cv;

unsigned char** randomCreator(int);
int dist(unsigned char, unsigned char, unsigned char, unsigned char, unsigned char, unsigned char);
unsigned char** clustring(Mat, int, int, unsigned char**);
void connected_Component_Segmentation(Mat, unsigned char**, int);
void segmentation(Mat, unsigned char**, unsigned char**, int);
void changeLabels(int, int, unsigned short**, unsigned short, unsigned short);

int main() {

	srand(time(NULL));

	Mat modifiedImg = imread("image.jpg", IMREAD_COLOR);
	Mat clusteredImg;
	modifiedImg.copyTo(clusteredImg);
	int rows = modifiedImg.rows;
	int cols = modifiedImg.cols;
	int k = 4;
	int treshold = 20;
	int i = -1;
	int j = -1;

	printf("Enter a k value = ");
	scanf_s("%d", &k);
	printf("Enter a treshold value = ");
	scanf_s("%d", &treshold);
	printf("Process Started..\n");

	unsigned char** randomPixel = randomCreator(k);

	unsigned char** labelMat = clustring(modifiedImg, k, treshold, randomPixel);

	printf("After k-means clustring : \n");
	for (i = 150; i < 200; i++) {
		for (j = 150; j < 200; j++) {
			printf("%u ", labelMat[i][j]);
		}
		printf("\n");
	}

	
	segmentation(clusteredImg, labelMat, randomPixel, k);
	

	connected_Component_Segmentation(modifiedImg, labelMat, k);
	imwrite("Connected_Component_Segmantation.png", modifiedImg);
	imwrite("Clustring.png", clusteredImg);
	printf("Process Successful..\n");
	modifiedImg.release();
	clusteredImg.release();
	return 0;
}


// FUNCTIONS
unsigned char** randomCreator(int k) {

	int i = -1;
	unsigned char** randArray = (unsigned char**)malloc(sizeof(unsigned char) * 3);

	for (i = 0; i < 3; i++) {
		randArray[i] = (unsigned char*)malloc(sizeof(unsigned char) * k);
	}

	printf("Selected Colors : \n");
	for (i = 0; i < k; i++) {
		randArray[0][i] = ((i * 17) % 256); // blue value
		randArray[1][i] = ((i * 53) % 256); // green value
		randArray[2][i] = ((i * 31) % 256); // red value
		printf("b=%u - g=%u - r=%u\n", randArray[0][i], randArray[1][i], randArray[2][i]);
	}

	return randArray;
}

int dist(unsigned char a1, unsigned char b1, unsigned char c1, unsigned char a2, unsigned char b2, unsigned char c2) {
	return sqrt((a1 - a2) * (a1 - a2) + (b1 - b2) * (b1 - b2) + (c1 - c2) * (c1 - c2));
}

unsigned char** clustring(Mat img, int k, int treshold, unsigned char** randomPixel) {

	int control = treshold + 1; // en az 1 kere iþleme girmesini istediðimiz için tresholddan büyük bir deðer verdik.
	int minDist; // o anki pixelin hangi Mü deðerine yakýn olduðunu bulmak için kullandýðýmýz deðiþken.
	unsigned char minDistIndex; // anlýk pixelin en yakýn olduðu Mü deðerinin indisi.
	int distance; // anlýk uzaklýk
	int* count;
	int i = 0, j = 0, l = 1;
	count = (int*)malloc(sizeof(int) * k); // her Mü deðeri için kaç adet yakýnsayan pixel olduðunu tutan sayaç.
	int** sum = (int**)malloc(sizeof(int*) * 3); // 1. indis mavi - 2. indis yeþil - 3. indis kýrmýzý toplamlarý tutacak.
	for (int i = 0; i < 3; i++)
		sum[i] = (int*)malloc(sizeof(int) * k);

	// label matrisi oluþturulmasý---
	unsigned char** labelMat;
	labelMat = (unsigned char**)malloc(sizeof(unsigned char*) * img.rows);
	for (i = 0; i < img.rows; i++) {
		labelMat[i] = (unsigned char*)malloc(sizeof(unsigned char) * img.cols);
	}
	// ------------------------------

	while (treshold < control) {

		for (i = 0; i < k; i++) {
			sum[0][i] = 0;
			sum[1][i] = 0;
			sum[2][i] = 0;
			count[i] = 0;
		}
		for (i = 0; i < img.rows; i++) {
			for (j = 0; j < img.cols; j++) {

				minDist = dist(img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2],
					randomPixel[0][0], randomPixel[1][0], randomPixel[2][0]);

				minDistIndex = 0;

				for (l = 1; l < k; l++) {

					distance = dist(img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2],
						randomPixel[0][l], randomPixel[1][l], randomPixel[2][l]);

					if (distance < minDist) {

						minDist = distance;
						minDistIndex = l;
					}
				}

				sum[0][minDistIndex] += img.at<Vec3b>(i, j)[0];
				sum[1][minDistIndex] += img.at<Vec3b>(i, j)[1];
				sum[2][minDistIndex] += img.at<Vec3b>(i, j)[2];
				count[minDistIndex]++;
				labelMat[i][j] = minDistIndex;

			}
		}

		control = 0;
		for (i = 0; i < k; i++) {
			if (count[i] != 0) {
				sum[0][i] = sum[0][i] / count[i];
				sum[1][i] = sum[1][i] / count[i];
				sum[2][i] = sum[2][i] / count[i];
				control += dist(randomPixel[0][i], randomPixel[1][i], randomPixel[2][i], sum[0][i], sum[1][i], sum[2][i]);
			}
		}
	
		for (i = 0; i < k; i++) {
			if (count[i] != 0) {
				randomPixel[0][i] = sum[0][i];
				randomPixel[1][i] = sum[1][i];
				randomPixel[2][i] = sum[2][i];
			}
		}

	}

	free(sum);
	free(count);

	return labelMat;
}

void segmentation(Mat img, unsigned char** labelMat, unsigned char** randomPixel, int k) {

	int i = -1, j = -1;
	for (i = 0; i < img.rows; i++)
		for (j = 0; j < img.cols; j++) {
			img.at<Vec3b>(i, j)[0] = randomPixel[0][ labelMat[i][j] ];
			img.at<Vec3b>(i, j)[1] = randomPixel[1][ labelMat[i][j] ];
			img.at<Vec3b>(i, j)[2] = randomPixel[2][ labelMat[i][j] ];
		}
}

void connected_Component_Segmentation(Mat img, unsigned char** labelMat, int k) {
	
	int i = -1, j = -1, count = 1;

	unsigned short** newLabel = (unsigned short**)calloc((img.rows), sizeof(unsigned short*));
	for (i = 0; i < img.rows; i++)
		newLabel[i] = (unsigned short*)calloc((img.cols), sizeof(unsigned short));
	
	newLabel[0][0] = 1;

	for (i = 1; i < img.cols; i++) {
		if (labelMat[0][i] == labelMat[0][i - 1]) 
			newLabel[0][i] = newLabel[0][i - 1];
		else
			newLabel[0][i] = ++count;
	}

	for (i = 1; i < img.rows; i++) {
		
		if (labelMat[i][0] == labelMat[i - 1][0]) 
			newLabel[i][0] = newLabel[i - 1][0];
		else if (labelMat[i][0] == labelMat[i - 1][1])
			newLabel[i][0] = newLabel[i - 1][1];
		else 
			newLabel[i][0] = ++count;
		
		for (j = 1; j < img.cols - 1; j++) {
			if (labelMat[i][j] == labelMat[i][j - 1]) {
				newLabel[i][j] = newLabel[i][j - 1];
				if (labelMat[i][j] == labelMat[i - 1][j + 1])
					if(newLabel[i][j] != newLabel[i - 1][j + 1])
						changeLabels(i, img.cols, newLabel, newLabel[i - 1][j + 1], newLabel[i][j]);
			}
			else if (labelMat[i][j] == labelMat[i - 1][j - 1]) {
				newLabel[i][j] = newLabel[i - 1][j - 1];
				if (labelMat[i][j] == labelMat[i - 1][j + 1]) 
					if (newLabel[i][j] != newLabel[i - 1][j + 1])
						changeLabels(i, img.cols, newLabel, newLabel[i - 1][j + 1], newLabel[i][j]);
			}
			else if (labelMat[i][j] == labelMat[i - 1][j])
				newLabel[i][j] = newLabel[i - 1][j];
			else if (labelMat[i][j] == labelMat[i - 1][j + 1])
				newLabel[i][j] = newLabel[i - 1][j + 1];
			else
				newLabel[i][j] = ++count;
		}

		if (labelMat[i][j] == labelMat[i - 1][j])
			newLabel[i][j] = newLabel[i - 1][j];
		else if (labelMat[i][j] == labelMat[i - 1][j - 1])
			newLabel[i][j] = newLabel[i - 1][j - 1];
		else if (labelMat[i][j] == labelMat[i][j - 1])
			newLabel[i][j] = newLabel[i][j - 1];
		else 
			newLabel[i][j] = ++count;
	}

	for (i = 0; i < img.rows; i++) 
		for (j = 0; j < img.cols; j++) {
			img.at<Vec3b>(i, j)[0] = (newLabel[i][j] * 17) % 256;
			img.at<Vec3b>(i, j)[1] = (newLabel[i][j] * 51) % 256;
			img.at<Vec3b>(i, j)[2] = (newLabel[i][j] * 31) % 256;
		}

}
void changeLabels(int row, int col, unsigned short** newLabel, unsigned short changing, unsigned short changer) {
	int i = -1, j = -1;
	for (i = 0; i <= row; i++)
		for (j = 0; j < col; j++)
			if (newLabel[i][j] == changing)
				newLabel[i][j] = changer;
}
