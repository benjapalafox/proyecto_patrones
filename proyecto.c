/************************************* Include Files *********************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

/************************************* Constant Definitions **************************************/
#define TRAINING_FILE_IMAGES  "mnist/train-images.idx3-ubyte"
#define TRAINING_FILE_LABELS  "mnist/train-labels.idx1-ubyte"
#define TESTING_FILE_IMAGES   "mnist/t10k-images.idx3-ubyte"
#define TESTING_FILE_LABELS   "mnist/t10k-labels.idx1-ubyte"
#define RESULTS_FILE          "mnist_results.txt"

#define MNIST_TRAINING_SIZE   60000
#define MNIST_TESTING_SIZE    10000
#define MNIST_IMAGE_WIDTH     28
#define MNIST_IMAGE_HEIGHT    28
#define MNIST_IMAGE_SIZE      784

#define MNIST_TRAINING_SET    0
#define MNIST_TESTING_SET     1

#define K_NN                  3
/************************************* Type Definitions ******************************************/
typedef struct
{
	long int size;
	unsigned char *data;
} file_t;

typedef struct
{
	long int size;
	unsigned char* data;
} bmp_image_data_t;

typedef struct
{
	long int width;
	long int height;
	unsigned char* data;
} mnist_image_t;

typedef struct
{
	long int image_count;
	unsigned char* label;
	mnist_image_t** image;
} mnist_image_list_t;

typedef struct
{
	unsigned char label;
	double distance;
} neighbor_t;

typedef struct neighbor
{
	long int neighbor_count;
	neighbor_t** neighbor;
} neighbor_list_t;

/************************************* Function Prototypes ***************************************/
file_t* file_alloc(long int size);
void file_free(file_t* file);
file_t* file_read(char* file_name);

bmp_image_data_t* bmp_image_data_alloc(long int bmp_image_size);
void bmp_image_data_free(bmp_image_data_t* bmp_image);
bmp_image_data_t* bmp_image_data_get(long int width, long int height, unsigned char* data);

mnist_image_t* mnist_image_alloc(long int width, long int height);
void mnist_image_free(mnist_image_t* image);
void mnist_image_fprintf(char *file_name, mnist_image_t* image);

mnist_image_list_t* mnist_image_list_alloc(long int image_count);
void mnist_image_list_free(mnist_image_list_t* list);
mnist_image_list_t* mnist_image_list_get(file_t* f_images, file_t* f_labels, char set);
void mnist_image_list_to_bmp(mnist_image_list_t* list, char set);

double k_NN_image_distance(mnist_image_t* image_1, mnist_image_t * image_2);
unsigned char k_NN_classifier(mnist_image_list_t* training_list, mnist_image_t* image);
void k_NN_classifier_test(char* file_name, mnist_image_list_t* training_list, mnist_image_list_t* testing_list);

/************************************* Variable Definitions **************************************/

/************************************* Function Declarations *************************************/

int main(int argc, char** argv)
{
	int world_rank;
	int world_size;
	file_t* f_images;
	file_t* f_labels;
	mnist_image_list_t* training_list;
	mnist_image_list_t* testing_list;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_rank == 0)
	{
		printf("Loading images.\n");
	}

	f_images = file_read(TRAINING_FILE_IMAGES);
	f_labels = file_read(TRAINING_FILE_LABELS);
	training_list = mnist_image_list_get(f_images, f_labels, MNIST_TRAINING_SET);

	f_images = file_read(TESTING_FILE_IMAGES);
	f_labels = file_read(TESTING_FILE_LABELS);
	testing_list = mnist_image_list_get(f_images, f_labels, MNIST_TESTING_SET);

	if (world_rank == 0)
	{
		mnist_image_list_to_bmp(training_list, MNIST_TRAINING_SET);
		mnist_image_list_to_bmp(testing_list, MNIST_TESTING_SET);
		
		printf("Images loaded.\n");
		printf("Classifier test started.\n");
	}

	k_NN_classifier_test(RESULTS_FILE, training_list, testing_list);

	if (world_rank == 0)
	{
		printf("Classifier test finished.\n");
		printf("Check %s file.\n", RESULTS_FILE);
	}

	mnist_image_list_free(testing_list);
	mnist_image_list_free(training_list);

	MPI_Finalize();

	return 0;
}

/*************************************************************************************************/
file_t* file_alloc(long int size)
{
	file_t* file;

	if (size <= 0)
	{
		printf("Failed to allocate file. Size must be a positive integer.\n");
		exit(0);
	}

	file = malloc(sizeof(file_t));
	if (file == NULL)
	{
		printf("Failed to allocate file. Failed to allocate space for file struct.\n");
		exit(0);
	}

	file->data = malloc(sizeof(char) * size);
	if (file->data == NULL)
	{
		printf("Failed to allocate file. Failed to allocate space for file data.\n");
		exit(0);
	}

	return file;
}

/*************************************************************************************************/
void file_free(file_t* file)
{
	if (file == NULL)
	{
		return;
	}
	if (file->data != NULL)
	{
		free(file->data);
	}
	free(file);
}

/*************************************************************************************************/
file_t* file_read(char* file_name)
{
	long int file_size;
	FILE* fp = NULL;
	file_t* file;

	if (file_name == NULL)
	{
		printf("Failed to read file. File name can't be NULL.\n");
		exit(0);
	}

	fp = fopen(file_name, "rb");
	if (fp == NULL)
	{
		printf("Failed to read file. Failed to open file.\n");
		exit(0);
	}

	if (fseek(fp, 0, SEEK_END) != 0)
	{
		fclose(fp);
		printf("Failed to read file. Failed to set end position.\n");
		exit(0);
	}

	file_size = ftell(fp);
	if (file_size == -1)
	{
		fclose(fp);
		printf("Failed to read file. Read position error.\n");
		exit(0);
	}
	if (file_size == 0)
	{
		fclose(fp);
		printf("Failed to read file. Empty file.\n");
		exit(0);
	}

	if (fseek(fp, 0, SEEK_SET) != 0)
	{
		fclose(fp);
		printf("Failed to read file. Failed to set beginning position.\n");
		exit(0);
	}

	file = file_alloc(file_size);

	file->size = fread(file->data, 1, file_size, fp);
	if (file->size != file_size)
	{
		file_free(file);
		fclose(fp);
		printf("Failed to read file. Failed to read all file.\n");
		exit(0);
	}

	fclose(fp);
	return file;
}

/*************************************************************************************************/
bmp_image_data_t* bmp_image_data_alloc(long int bmp_image_size)
{
	bmp_image_data_t* bmp_image;

	bmp_image = malloc(sizeof(bmp_image_data_t));
	if (bmp_image == NULL)
	{
		exit(0);
	}

	bmp_image->data = malloc(sizeof(unsigned char) * bmp_image_size);
	if (bmp_image->data == NULL)
	{
		exit(0);
	}

	bmp_image->size = bmp_image_size;

	return bmp_image;
}

/*************************************************************************************************/
void bmp_image_data_free(bmp_image_data_t* bmp_image)
{
	if (bmp_image == NULL)
	{
		return;
	}
	if (bmp_image->data != NULL)
	{
		free(bmp_image->data);
	}

	free(bmp_image);
}

/*************************************************************************************************/
bmp_image_data_t* bmp_image_data_get(long int width, long int height, unsigned char* data)
{
	unsigned char bmp_file_header[14] = { 66, 77, 70, 7, 0, 0, 0, 0, 0, 0, 54, 4, 0, 0 };
	unsigned char bmp_dib_header[40] = { 40, 0, 0, 0, 28, 0, 0, 0, 28, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	long int i;
	long int j;
	long int k;
	bmp_image_data_t* bmp_image;

	bmp_image = bmp_image_data_alloc(1862);

	for (i = 0; i < 14; i++)
	{
		bmp_image->data[i] = bmp_file_header[i];
	}

	for (i = 14, j = 0; i < 54; i++, j++)
	{
		bmp_image->data[i] = bmp_dib_header[j];
	}

	for (i = 54, j = 0; i < 1078; i += 4, j++)
	{
		bmp_image->data[i] = j;
		bmp_image->data[i + 1] = j;
		bmp_image->data[i + 2] = j;
		bmp_image->data[i + 3] = 0;
	}

	for (j = 27, i = 1078; j >= 0; j--)
	{
		for (k = 0; k < 28; k++, i++)
		{
			bmp_image->data[i] = 255 - data[28 * j + k];
		}
	}

	return bmp_image;
}

/*************************************************************************************************/
mnist_image_t* mnist_image_alloc(long int width, long int height)
{
	mnist_image_t* image;

	if (width <= 0)
	{
		printf("Failed to allocate image. Image width must be a positive integer.\n");
		exit(0);
	}

	if (height <= 0)
	{
		printf("Failed to allocate image. Image height must be a positive integer.\n");
		exit(0);
	}

	image = malloc(sizeof(mnist_image_t));
	if (image == NULL)
	{
		printf("Failed to allocate image. Failed to allocate space for image struct.\n");
		exit(0);
	}

	image->data = malloc(sizeof(unsigned char) * (width * height));
	if (image->data == NULL)
	{
		printf("Failed to allocate image. Failed to allocate space for image.\n");
		exit(0);
	}

	image->width = width;
	image->height = height;

	return image;
}

/*************************************************************************************************/
void mnist_image_free(mnist_image_t* image)
{
	if (image == NULL)
	{
		return;
	}

	if (image->data != NULL)
	{
		free(image->data);
	}

	free(image);
}

/*************************************************************************************************/
void mnist_image_fprintf(char *file_name, mnist_image_t* image)
{
	long int size;
	FILE* fp = NULL;
	bmp_image_data_t* bmp_image;

	if (file_name == NULL)
	{
		printf("Failed to save image to file. File name can't be NULL.\n");
		exit(0);
	}

	if (image == NULL)
	{
		printf("Failed to save image to file. Image can't be NULL.\n");
		exit(0);
	}

	bmp_image = bmp_image_data_get(image->width, image->height, image->data);

	fp = fopen(file_name, "w");
	if (fp == NULL)
	{
		printf("Failed to save image to file. Failed to open file.\n");
		exit(0);
	}

	size = fwrite(bmp_image->data, sizeof(unsigned char), bmp_image->size, fp);

	if (size != bmp_image->size)
	{
		printf("Failed to save image to file. Failed write image to file.\n");
		exit(0);
	}

	fclose(fp);
}

/*************************************************************************************************/
mnist_image_list_t* mnist_image_list_alloc(long int image_count)
{
	long int i;
	mnist_image_list_t* list;

	if (image_count <= 0)
	{
		printf("Failed to allocate image list. Image count must be a positive integer.\n");
		exit(0);
	}

	list = malloc(sizeof(mnist_image_list_t));
	if (list == NULL)
	{
		printf("Failed to allocate image list. Failed to allocate space for image list struct.\n");
		exit(0);
	}

	list->label = malloc(sizeof(unsigned char) * image_count);
	if (list->label == NULL)
	{
		printf("Failed to allocate image list. Failed to allocate space for image list labels.\n");
		exit(0);
	}

	list->image = malloc(sizeof(mnist_image_t *) * image_count);
	if (list->image == NULL)
	{
		printf("Failed to allocate image list. Failed to allocate space for image list images.\n");
		exit(0);
	}

	for (i = 0; i < image_count; i++)
	{
		list->image[i] = mnist_image_alloc(MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT);
	}

	list->image_count = image_count;

	return list;
}

/*************************************************************************************************/
void mnist_image_list_free(mnist_image_list_t* list)
{
	long int i;

	if (list == NULL)
	{
		return;
	}

	if (list->label != NULL)
	{
		free(list->label);
	}

	if (list->image != NULL)
	{
		for (i = 0; i < list->image_count; i++)
		{
			mnist_image_free(list->image[i]);
		}
		free(list->image);
	}
	free(list);
}

/*************************************************************************************************/
mnist_image_list_t* mnist_image_list_get(file_t* f_images, file_t* f_labels, char set)
{
	long int elements_number;
	long int image_rows_number;
	long int image_cols_number;
	long int i;
	long int j;
	long int l;
	long int p;
	mnist_image_list_t* list;

	//Start validation
	if (f_images == NULL)
	{
		printf("Failed to get image set. Images file is NULL.\n");
		exit(0);
	}
	if (f_labels == NULL)
	{
		printf("Failed to get image set. Labels file is NULL.\n");
		exit(0);
	}
	if ((set != MNIST_TRAINING_SET) && (set != MNIST_TESTING_SET))
	{
		printf("Failed to get image set. Set must be either MNIST_TRAINING_SET or MNIST_TESTING_SET.\n");
		exit(0);
	}

	if (f_images->size < 15 || f_labels->size < 7)
	{
		printf("Failed to get image set. One or both files have a corrupt header.\n");
		exit(0);
	}
	if ((f_images->data[0] != 0) || (f_images->data[1] != 0) || (f_images->data[2] != 8) || (f_images->data[3] != 3))
	{
		printf("Failed to get image set. Images file's magic number is incorrect.\n");
		exit(0);
	}

	if ((f_labels->data[0] != 0) || (f_labels->data[1] != 0) || (f_labels->data[2] != 8) || (f_labels->data[3] != 1))
	{
		printf("Failed to get image set. Labels file's magic number is incorrect.\n");
		exit(0);
	}

	if ((f_images->data[4] != f_labels->data[4]) || (f_images->data[5] != f_labels->data[5]) || (f_images->data[6] != f_labels->data[6]) || (f_images->data[7] != f_labels->data[7]))
	{
		printf("Failed to get image set. Images file's number of images does not match labels file's number of items.\n");
		exit(0);
	}

	for (i = 4, elements_number = 0, image_rows_number = 0, image_cols_number = 0; i < 8; i++)
	{
		elements_number += f_images->data[i];
		image_rows_number += f_images->data[i + 4];
		image_cols_number += f_images->data[i + 8];

		if (i < 7)
		{
			elements_number <<= 8;
			image_rows_number <<= 8;
			image_cols_number <<= 8;
		}
	}

	if (((set == MNIST_TRAINING_SET) && (elements_number != MNIST_TRAINING_SIZE)) || ((set == MNIST_TESTING_SET) && (elements_number != MNIST_TESTING_SIZE)))
	{
		printf("Failed to get image set. Number of elements is different than expected.\n");
		exit(0);
	}

	if ((image_rows_number != MNIST_IMAGE_HEIGHT))
	{
		printf("Failed to get image set. Images file's number rows is incorrect.\n");
		exit(0);
	}
	if ((image_cols_number != MNIST_IMAGE_WIDTH))
	{
		printf("Failed to get image set. Images file's number columns is incorrect.\n");
		exit(0);
	}

	if (f_images->size != (MNIST_IMAGE_SIZE * elements_number + 16))
	{
		printf("Failed to get image set. Size of images file is incorrect.\n");
		exit(0);
	}

	if (f_labels->size != (elements_number + 8))
	{
		printf("Failed to get image set. Size of labels file is incorrect.\n");
		exit(0);
	}

	//Both files look good

	list = mnist_image_list_alloc(elements_number);

	for (i = 0, p = 16, l = 8; i < elements_number; i++, l++)
	{
		for (j = 0; j < MNIST_IMAGE_SIZE; j++, p++)
		{
			list->image[i]->data[j] = f_images->data[p];
		}
		list->label[i] = f_labels->data[l];
	}

	file_free(f_images);
	file_free(f_labels);

	return list;
}

/*************************************************************************************************/
void mnist_image_list_to_bmp(mnist_image_list_t* list, char set)
{
	char name[37];
	long int i;

	for (i = 0; i < list->image_count; i++)
	{
		if (set == MNIST_TRAINING_SET)
		{
			sprintf(name, "training/label_%u_element_%05ld.bmp", list->label[i], i);
		}
		else
		{
			sprintf(name, "testing/label_%u_element_%05ld.bmp", list->label[i], i);
		}

		mnist_image_fprintf(name, list->image[i]);
	}
}

/*************************************************************************************************/
double k_NN_image_distance(mnist_image_t* image_1, mnist_image_t * image_2)
{
	long int i;
	double tmp;
	double sum;

	for (i = 0, sum = 0; i < MNIST_IMAGE_SIZE; i++)
	{
		tmp = image_1->data[i] - image_2->data[i];
		sum += pow(tmp, 2);
	}

	return sqrt(sum);
}

/*************************************************************************************************/
unsigned char k_NN_classifier(mnist_image_list_t* training_list, mnist_image_t* image)
{
	long int k_neighbors = K_NN;
	long int i;
	long int j;
	long int best_label_count;
	unsigned char best_label;
	long int* neighbors_count;
	double distance;
	neighbor_t* tmp_neighbor;
	neighbor_list_t* neighbors_list;

	neighbors_list = malloc(sizeof(neighbor_list_t));
	if (neighbors_list == NULL)
	{
		printf("Failed to classify with k-nearest neighbors.\n");
		exit(0);
	}

	neighbors_list->neighbor = malloc(sizeof(neighbor_t *) * k_neighbors);
	if (neighbors_list->neighbor == NULL)
	{
		printf("Failed to classify with k-nearest neighbors.\n");
		exit(0);
	}

	for (i = 0; i < k_neighbors; i++)
	{
		neighbors_list->neighbor[i] = malloc(sizeof(neighbor_t));
		if (neighbors_list->neighbor[i] == NULL)
		{
			printf("Failed to classify with k-nearest neighbors.\n");
			exit(0);
		}
	}

	tmp_neighbor = malloc(sizeof(neighbor_t));
	if (tmp_neighbor == NULL)
	{
		printf("Failed to classify with k-nearest neighbors.\n");
		exit(0);
	}

	for (i = 0, neighbors_list->neighbor_count = 0; i < training_list->image_count; i++)
	{
		distance = k_NN_image_distance(training_list->image[i], image);

		if (neighbors_list->neighbor_count < k_neighbors)
		{
			neighbors_list->neighbor[neighbors_list->neighbor_count]->label = training_list->label[i];
			neighbors_list->neighbor[neighbors_list->neighbor_count]->distance = distance;
			neighbors_list->neighbor_count++;
		}
		else if (distance < neighbors_list->neighbor[k_neighbors - 1]->distance)
		{
			neighbors_list->neighbor[k_neighbors - 1]->label = training_list->label[i];
			neighbors_list->neighbor[k_neighbors - 1]->distance = distance;
			for (j = k_neighbors - 1; j > 0; j--)
			{
				if (neighbors_list->neighbor[j]->distance < neighbors_list->neighbor[j - 1]->distance)
				{
					tmp_neighbor->label = neighbors_list->neighbor[j - 1]->label;
					tmp_neighbor->distance = neighbors_list->neighbor[j - 1]->distance;

					neighbors_list->neighbor[j - 1]->label = neighbors_list->neighbor[j]->label;
					neighbors_list->neighbor[j - 1]->distance = neighbors_list->neighbor[j]->distance;

					neighbors_list->neighbor[j]->label = tmp_neighbor->label;
					neighbors_list->neighbor[j]->distance = tmp_neighbor->distance;
				}
			}
		}
	}

	free(tmp_neighbor);

	neighbors_count = malloc(sizeof(long int) * 10);
	if (neighbors_count == NULL)
	{
		printf("Failed to classify with k-nearest neighbors. Failed to allocate space for neighbor count.\n");
		exit(0);
	}

	for (i = 0; i < 10; i++)
	{
		neighbors_count[i] = 0;
	}

	for (i = 0; i < k_neighbors; i++)
	{
		neighbors_count[neighbors_list->neighbor[i]->label]++;
	}

	for (i = 0; i < k_neighbors; i++)
	{
		free(neighbors_list->neighbor[i]);
	}
	free(neighbors_list->neighbor);
	free(neighbors_list);

	for (i = 0; i < 10; i++)
	{
		if (i == 0 || neighbors_count[i] > best_label_count)
		{
			best_label_count = neighbors_count[i];
			best_label = i;
		}
	}

	free(neighbors_count);

	return best_label;
}

/*************************************************************************************************/
void k_NN_classifier_test(char* file_name, mnist_image_list_t* training_list, mnist_image_list_t* testing_list)
{
	unsigned char label;
	char name[36];
	long int i;
	long int j;
	long int interval;
	long int k_neighbors = K_NN;
	double errors;
	double counter;
	double errors_r;
	double counter_r;
	double error;
	FILE *fp = NULL;

	int world_rank;
	int world_size;

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	interval = testing_list->image_count / world_size;
	j = interval * (world_rank + 1);

	for (i = (interval * world_rank), errors = 0, counter = 0; i < j; i++)
	{
		if (world_rank == 0)
		{
			printf("\r%04.1f%%", (100 * (double)i) / (double)j);
			fflush(stdout);
		}
		label = k_NN_classifier(training_list, testing_list->image[i]);
		if (label != testing_list->label[i])
		{
			errors += 1.0;
			sprintf(name, "errors/element_%05ld_labeled_%u.bmp", i, label);
			mnist_image_fprintf(name, testing_list->image[i]);
		}
		counter += 1.0;
	}

	if (world_rank == 0)
	{
		printf("\n");
		printf("Receiving data from processes.\n");
		for (i = 1; i < world_size; i++)
		{
			MPI_Recv(&errors_r, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&counter_r, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			errors += errors_r;
			counter += counter_r;
		}
	}
	else
	{
		MPI_Send(&errors, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&counter, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	}

	if (world_rank == 0)
	{
		error = 100.0 * errors / counter;

		fp = fopen(file_name, "a");
		if (fp == NULL)
		{
			printf("Failed to print k-nearest neighbors classifier test results to file. Error opening file.\n");
			printf("\n");
			printf("k = %li\nImages tested = %d\nCorrectly classified images  = %d\nError rate = %f%%\n", k_neighbors, (int)counter, (int)(counter - errors), error);
			printf("\n");
		}
		else
		{
			fprintf(fp, "\n");
			fprintf(fp, "k = %li\nImages tested = %d\nCorrectly classified images  = %d\nError rate = %f%%\n", k_neighbors, (int)counter, (int)(counter - errors), error);
			fprintf(fp, "/*************************************************************************************************/\n");
			fprintf(fp, "\n");
			fclose(fp);
		}
	}
}
