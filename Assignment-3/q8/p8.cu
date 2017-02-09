#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <cuda_runtime.h>

FILE*forg = fopen("/lena_gray.bmp", "rb");
FILE*fsz = fopen("/i_bur.bmp", "wb");  


__global__ void GaussianBlur(unsigned int *B,unsigned int *G,unsigned int *R, int numberOfPixels, unsigned int width, int *B_new, int *G_new, int *R_new)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numberOfPixels){
		//printf("%d\n",index);
		return;
	}

	int mask[] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	int s = mask[0] + mask[1] + mask[2] + mask[3] + mask[4] + mask[5] + mask[6] + mask[7] + mask[8];

	if (index < width){ // dolny rzad pikseli
		if (index == 0){ //lewy dolny rog
			s = mask[4] + mask[1] + mask[2] + mask[5];
			B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width + 1] * mask[2] + B[index + 1] * mask[5]) / s);
			G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width + 1] * mask[2] + G[index + 1] * mask[5]) / s);
			R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width + 1] * mask[2] + R[index + 1] * mask[5]) / s);
			return;
		}

		if (index == width - 1){//prawy dolny rog
			s = mask[4] + mask[0] + mask[1] + mask[3];
			B_new[index] = (B[index] * mask[4] + B[index + width - 1] * mask[0] + B[index + width] * mask[1] + B[index - 1] * mask[3]);
			G_new[index] = (G[index] * mask[4] + G[index + width - 1] * mask[0] + G[index + width] * mask[1] + G[index - 1] * mask[3]);
			R_new[index] = (R[index] * mask[4] + R[index + width - 1] * mask[0] + R[index + width] * mask[1] + R[index - 1] * mask[3]);
			return;
		}
		//reszta pikseli w dolnym rzedzie
		s = mask[4] + mask[1] + mask[2] + mask[5] + mask[0] + mask[3];
		B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width + 1] * mask[2] + B[index + 1] * mask[5] + B[index + width - 1] * mask[0] + B[index - 1] * mask[3]) / s);
		R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width + 1] * mask[2] + R[index + 1] * mask[5] + R[index + width - 1] * mask[0] + R[index - 1] * mask[3]) / s);
		G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width + 1] * mask[2] + G[index + 1] * mask[5] + G[index + width - 1] * mask[0] + G[index - 1] * mask[3]) / s);

		return;
	}
	if (index >= numberOfPixels - width){ //gorny rzad pikseli

		if (index == numberOfPixels - width){ //lewy gorny rog
			s = mask[4] + mask[5] + mask[7] + mask[8];
			B_new[index] = (int)((B[index] * mask[4] + B[index + 1] * mask[5] + B[index - width] * mask[7] + B[index - width + 1] * mask[8]) / s);
			G_new[index] = (int)((G[index] * mask[4] + G[index + 1] * mask[5] + G[index - width] * mask[7] + G[index - width + 1] * mask[8]) / s);
			R_new[index] = (int)((R[index] * mask[4] + R[index + 1] * mask[5] + R[index - width] * mask[7] + R[index - width + 1] * mask[8]) / s);
			return;
		}

		if (index == numberOfPixels - 1){ //prawy gorny rog
			s = mask[4] + mask[3] + mask[6] + mask[7];
			B_new[index] = (int)((B[index] * mask[4] + B[index - 1] * mask[3] + B[index - width - 1] * mask[6] + B[index - width] * mask[7]) / s);
			G_new[index] = (int)((G[index] * mask[4] + G[index - 1] * mask[3] + G[index - width - 1] * mask[6] + G[index - width] * mask[7]) / s);
			R_new[index] = (int)((R[index] * mask[4] + R[index - 1] * mask[3] + R[index - width - 1] * mask[6] + R[index - width] * mask[7]) / s);
			return;
		}

		s = mask[4] + mask[3] + mask[5] + mask[6] + mask[7] + mask[8];
		B_new[index] = (int)((B[index] * mask[4] + B[index - 1] * mask[3] + B[index - width - 1] * mask[6] + B[index - width] * mask[7] + B[index + 1] * mask[5] + B[index - width] * mask[8]) / s);
		R_new[index] = (int)((R[index] * mask[4] + R[index - 1] * mask[3] + R[index - width - 1] * mask[6] + R[index - width] * mask[7] + R[index + 1] * mask[5] + R[index - width] * mask[8]) / s);
		G_new[index] = (int)((G[index] * mask[4] + G[index - 1] * mask[3] + G[index - width - 1] * mask[6] + G[index - width] * mask[7] + G[index + 1] * mask[5] + G[index - width] * mask[8]) / s);
		return;
	}
	if (index % width == 0){ //lewa sciana
		s = mask[4] + mask[1] + mask[2] + mask[5] + mask[8] + mask[7];
		B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width + 1] * mask[2] + B[index + 1] * mask[5] + B[index - width + 1] * mask[8] + B[index - width]) / s);
		G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width + 1] * mask[2] + G[index + 1] * mask[5] + G[index - width + 1] * mask[8] + G[index - width]) / s);
		R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width + 1] * mask[2] + R[index + 1] * mask[5] + R[index - width + 1] * mask[8] + R[index - width]) / s);
		return;
	}
	if (index % width == width - 1){ //prawa sciana
		s = mask[4] + mask[1] + mask[0] + mask[3] + mask[6] + mask[7];
		B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width - 1] * mask[0] + B[index - 1] * mask[3] + B[index - width - 1] * mask[6] + B[index - width] * mask[7]) / s);
		R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width - 1] * mask[0] + R[index - 1] * mask[3] + R[index - width - 1] * mask[6] + R[index - width] * mask[7]) / s);
		G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width - 1] * mask[0] + G[index - 1] * mask[3] + G[index - width - 1] * mask[6] + G[index - width] * mask[7]) / s);
		return;
	}


		int poz_1 = index - width - 1;
		int poz_2 = index - width;
		int poz_3 = index - width + 1;
		int poz_4 = index - 1;
		int poz_5 = index;
		int poz_6 = index + 1;
		int poz_7 = index + width - 1;
		int poz_8 = index + width;
		int poz_9 = index + width + 1;

		B_new[index] = (int)(((B[poz_1] * mask[0]) + (B[poz_2] * mask[1]) + (B[poz_3] * mask[2]) + (B[poz_4] * mask[3]) + (B[poz_5] * mask[4]) + (B[poz_6] * mask[5]) + (B[poz_7] * mask[6]) + (B[poz_8] * mask[7]) + (B[poz_9] * mask[8])) / s);
		G_new[index] = (int)(((G[poz_1] * mask[0]) + (G[poz_2] * mask[1]) + (G[poz_3] * mask[2]) + (G[poz_4] * mask[3]) + (G[poz_5] * mask[4]) + (G[poz_6] * mask[5]) + (G[poz_7] * mask[6]) + (G[poz_8] * mask[7]) + (G[poz_9] * mask[8])) / s);
		R_new[index] = (int)(((R[poz_1] * mask[0]) + (R[poz_2] * mask[1]) + (R[poz_3] * mask[2]) + (R[poz_4] * mask[3]) + (R[poz_5] * mask[4]) + (R[poz_6] * mask[5]) + (R[poz_7] * mask[6]) + (R[poz_8] * mask[7]) + (R[poz_9] * mask[8])) / s);


}


struct FileHeader {
	unsigned short bfType;
	unsigned int bfSize;
	short bfReserved1;
	short bfReserved2;
	short bfOffBits;
};
FileHeader File;

// Structure definition to store image info
struct PictureHeader {
	unsigned int biSize;
	unsigned int biWidth;
	unsigned int biHeight;
	short biPlanes;
	short biBitCount;
	int biCompression;
	int biSizeImage;
	int biXPelsPerMeter;
	int biYPelsPerMeter;
	int biClrUsed;
	int biClrImportant;
};
PictureHeader Picture;

int main()
{

unsigned char *bitmapImage;
long int imageIdx=0, pixelIdx=0;
int *d_B_new, *d_G_new, *d_R_new;

int size;

int z;

fseek(forg, 6, SEEK_SET);
fread(&File.bfReserved1, sizeof(File.bfReserved1),1,forg);
printf("%d\n",File.bfReserved1);
fread(&File.bfReserved2, sizeof(File.bfReserved2),1,forg);
printf("%d\n",File.bfReserved2);
fread(&File.bfOffBits, sizeof(File.bfOffBits),1,forg);
printf("%d\n",File.bfOffBits);

fread(&Picture.biSize, sizeof(Picture.biSize),1,forg);
printf("%d\n",Picture.biSize);
fread(&Picture.biWidth, sizeof(Picture.biWidth),1,forg);
printf("%d\n",Picture.biWidth);
fread(&Picture.biHeight, sizeof(Picture.biHeight),1,forg);
printf("%d\n",Picture.biHeight);
fread(&Picture.biPlanes, sizeof(Picture.biPlanes),1,forg);
printf("%d\n",Picture.biPlanes);
fread(&Picture.biBitCount, sizeof(Picture.biBitCount),1,forg);
printf("%d\n",Picture.biBitCount);
fread(&Picture.biCompression, sizeof(Picture.biCompression),1,forg);
printf("%d\n",Picture.biCompression);
fread(&Picture.biSizeImage, sizeof(Picture.biSizeImage),1,forg);
printf("%d\n",Picture.biSizeImage);
fread(&Picture.biXPelsPerMeter, sizeof(Picture.biXPelsPerMeter),1,forg);
printf("%d\n",Picture.biXPelsPerMeter);
fread(&Picture.biYPelsPerMeter, sizeof(Picture.biYPelsPerMeter),1,forg);
printf("%d\n",Picture.biYPelsPerMeter);
fread(&Picture.biClrUsed, sizeof(Picture.biClrUsed),1,forg);
printf("%d\n",Picture.biClrUsed);
fread(&Picture.biClrImportant, sizeof(Picture.biClrImportant),1,forg);
printf("%d\n\n",Picture.biClrImportant);

fseek(forg, 18, SEEK_SET);
fread(&Picture.biWidth, sizeof(Picture.biWidth),1,forg);
printf("%d\n",Picture.biWidth);
fread(&Picture.biHeight, sizeof(Picture.biHeight),1,forg);
printf("%d\n",Picture.biHeight);

size=Picture.biWidth*Picture.biHeight;

unsigned int R[Picture.biWidth*Picture.biHeight], G[Picture.biWidth*Picture.biHeight], B[Picture.biWidth*Picture.biHeight];
unsigned int *gpu_vector_R;
unsigned int *gpu_vector_G;
unsigned int *gpu_vector_B;

fseek(forg, 34, SEEK_SET);
fread(&Picture.biSizeImage, sizeof(Picture.biSizeImage),1,forg);
printf("%d\n",Picture.biSizeImage);

bitmapImage = (unsigned char*)malloc(Picture.biSizeImage);
fseek(forg, 54, SEEK_SET);
fread(bitmapImage,Picture.biSizeImage,1,forg);

//make sure bitmap image data was read
//    if (bitmapImage == NULL)
//    {
//        fclose(forg);
//        return NULL;
//    }

fseek(forg, 0, SEEK_SET);
	for (int i = 0; i < File.bfOffBits; i++)
	{
		z = fgetc(forg);
		fprintf(fsz,"%c", z);
	}

for (imageIdx = 0;imageIdx < Picture.biSizeImage;imageIdx+=3)
    {
	B[pixelIdx] = (unsigned int)(bitmapImage[imageIdx]);
        G[pixelIdx] = (unsigned int)(bitmapImage[imageIdx + 1]);
        R[pixelIdx] = (unsigned int)(bitmapImage[imageIdx + 2]);
        pixelIdx++;
    }

/*Allocating memory to vectors on GPU*/
	cudaMalloc((void **)&gpu_vector_B, size * sizeof(int));
	cudaMalloc((void **)&gpu_vector_G, size * sizeof(int));
	cudaMalloc((void **)&gpu_vector_R, size * sizeof(int));

	cudaMalloc((void **)&d_B_new, size * sizeof(int));
	cudaMalloc((void **)&d_G_new, size * sizeof(int));
	cudaMalloc((void **)&d_R_new, size * sizeof(int));


/*Copying vectors from CPU to GPU*/
	cudaMemcpy(gpu_vector_R, R, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_vector_G, G, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_vector_B, B, size * sizeof(int), cudaMemcpyHostToDevice);

GaussianBlur << < (size + 1023) / 1024, 1024 >> >(gpu_vector_B, gpu_vector_G, gpu_vector_R, size, Picture.biWidth, d_B_new, d_G_new, d_R_new);

cudaMemcpy(R, d_R_new, size * sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(G, d_G_new, size * sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(B, d_B_new, size * sizeof(int), cudaMemcpyDeviceToHost);

fseek(fsz, 54, SEEK_SET);

for (int i = 0; i < size; i++)
	{
		fprintf(fsz, "%c", (int)(B[i]));
		fprintf(fsz, "%c", (int)(G[i]));
		fprintf(fsz, "%c", (int)(R[i]));
}

fseek(fsz, 6, SEEK_SET);
fread(&File.bfReserved1, sizeof(File.bfReserved1),1,fsz);
printf("\n%d\n",File.bfReserved1);
fread(&File.bfReserved2, sizeof(File.bfReserved2),1,fsz);
printf("%d\n",File.bfReserved2);
fread(&File.bfOffBits, sizeof(File.bfOffBits),1,fsz);
printf("%d\n",File.bfOffBits);

fread(&Picture.biSize, sizeof(Picture.biSize),1,fsz);
printf("%d\n",Picture.biSize);
fread(&Picture.biWidth, sizeof(Picture.biWidth),1,fsz);
printf("%d\n",Picture.biWidth);
fread(&Picture.biHeight, sizeof(Picture.biHeight),1,fsz);
printf("%d\n",Picture.biHeight);
fread(&Picture.biPlanes, sizeof(Picture.biPlanes),1,fsz);
printf("%d\n",Picture.biPlanes);
fread(&Picture.biBitCount, sizeof(Picture.biBitCount),1,fsz);
printf("%d\n",Picture.biBitCount);
fread(&Picture.biCompression, sizeof(Picture.biCompression),1,fsz);
printf("%d\n",Picture.biCompression);
fread(&Picture.biSizeImage, sizeof(Picture.biSizeImage),1,fsz);
printf("%d\n",Picture.biSizeImage);
fread(&Picture.biXPelsPerMeter, sizeof(Picture.biXPelsPerMeter),1,fsz);
printf("%d\n",Picture.biXPelsPerMeter);
fread(&Picture.biYPelsPerMeter, sizeof(Picture.biYPelsPerMeter),1,fsz);
printf("%d\n",Picture.biYPelsPerMeter);
fread(&Picture.biClrUsed, sizeof(Picture.biClrUsed),1,fsz);
printf("%d\n",Picture.biClrUsed);
fread(&Picture.biClrImportant, sizeof(Picture.biClrImportant),1,fsz);
printf("%d\n",Picture.biClrImportant);

free(bitmapImage);

fclose(forg);
fclose(fsz);

	cudaFree(gpu_vector_R);
	cudaFree(gpu_vector_G);
	cudaFree(gpu_vector_B);

	cudaFree(d_B_new);
	cudaFree(d_G_new);
	cudaFree(d_R_new);

return (0);

}
