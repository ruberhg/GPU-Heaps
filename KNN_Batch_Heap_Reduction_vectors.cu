/*
* Copyright (C) 2022 Ricardo J. Barrientos (rbarrientos@ucm.cl)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <cuda.h>
#include <sys/resource.h>
#include <time.h>
#include <sys/time.h>

//Dimension of the elements
#define DIM 254

//The value of the K elements to be retrieved
#define TOPK 32

//Number of elements of the database
#define NE 1000000

//Number of threads of a warp in GPU
#define TAM_WARP 32

//Number of threads of a CUDA Block
#define TAM_MAX_BLOCK 352

//Q is the quantity of queries in each batch. It is limited by the memory
#define Q 3972 //with 95325 elements (vectors of dimension 254)
//#define Q 2979 //with 200000 elements (vectors of dimension 254)
//#define Q 1324 //with 500000 elements (vectors of dimension 254)
//#define Q 662 //with 999996 elements (vectors of dimension 254)
//#define Q 442 //with 1500000 elements (vectors of dimension 254)
//#define Q 331 //with 2000000 elements (vectors of dimension 254)


//Structure to keep the distance (to and from the query) and the index of the result
struct _Elem
{
  float dist;
  int ind;
};
typedef struct _Elem Elem;


//functions
__device__ void insertH(Elem *heap, Elem *elem, int *n_elem, int pitch, int id);
__device__ void extractH(Elem *heap, int *n_elem, int pitch, int id, Elem *eresult);
__device__ float topH(Elem *heap, int id);
__device__ void popush(Elem *heap, Elem *elem, int *n_elem, int pitch, int id);
__global__ void Batch_Heap_Reduction(float *DB_dev, int pitch_DB, Elem *heap, int pitch_H, float *QUERY_dev, int pitch_QUERY, Elem *arr_Dist, int pitch_Dist, int beginQ, float *res_final);
__device__ float distancia_trans(float *p1, int pitch_p1, int col_1, float *q);
int readdata(float *dato, FILE *file);


int N_QUERIES;


main(int argc, char *argv[])
{
   int i, N_ELEM, dimension, tam_elem, j;
   FILE *pf;
   float **vectores;
   float  real_time;
   struct timeval t1, t2;
   float *Elems, *QUERY_dev;
   float **consultas, *res_final, *res_final_H;
   int retorno, T_per_BLOCK, N_BLOQUES, contQ, cont;
   Elem *HEAPS_dev, *arr_res1, *arr_res1H, *arr_Dist;
   size_t pitch, pitch_H, pitch_Q, pitch_Dist;
   int *resT, *resTH;
   long long ED_total=0;
   double prom, prom_cont;


   if (argc != 6)
   {
      printf("\nExecute as: ./a.out file_BD file_queries DB_Size Queries_Size DIMENSION\n");
      return 0;
   }
   if ((pf = fopen(argv[1], "r")) == NULL)
   {
      printf("\nCannot open the file %s\n" ,argv[1]);
      return 0;
   }

   N_ELEM = atoi(argv[3]);
   N_QUERIES = atoi(argv[4]);
   dimension = atoi(argv[5]);

   if (dimension != DIM )
   {
     printf("\nERROR :: dimension != DIM\n");
     return 0;
   }

   printf("\nCant. Elementos=%d :: dimension=%d\n" , N_ELEM, dimension);
   fflush(stdout);
   if (N_ELEM != NE)
   {
     printf("\nERORR :: N_ELEM != NE\n");
     return 0;
   }
   if (N_ELEM < 512*32)
   {
     printf("\nERROR :: No enough database size to keep TOPK elements in each heap\n");
     return 0;
   }

   if (cudaSuccess != cudaMalloc((void **)&res_final, sizeof(float)*Q))
   {
     printf("\nERROR 21 :: cudaMalloc\n");
     cudaThreadExit();
     return 0;
   }
   res_final_H = (float *)malloc(sizeof(float)*Q);
   for (i=0; i<Q; i++)
   {
      res_final_H[i] = 0;
   }
   if (cudaSuccess != cudaMemset(res_final, 0, sizeof(float)*Q))
   {
       printf("\nERROR :: cudaMemset\n");
       cudaThreadExit();
       return 0;
   }

   //HEAPS_dev[TOPK][Q*512]
   if (cudaSuccess != cudaMallocPitch((void **)&HEAPS_dev, &pitch_H, sizeof(Elem)*Q*512, (size_t)TOPK))
   {
      printf("\nERROR 4 :: cudaMallocPitch :: Heaps_dev col=%lld :: row=%d\n", (long long)(sizeof(Elem)*Q*512), TOPK);
      cudaThreadExit();
      return 0;
   }

   //arr_Dist[Q][N_ELEM]
   if (cudaSuccess != cudaMallocPitch((void **)&arr_Dist, &pitch_Dist, N_ELEM*sizeof(Elem), (size_t)Q))
   {
      printf("\nERROR 41 :: cudaMallocPitch\n");
      cudaThreadExit();
      return 0;
   }

   vectores =(float **)malloc(sizeof(float *)*dimension);
   for (i=0; i<dimension; i++)
      vectores[i] = (float *)malloc(sizeof(float)*N_ELEM);

   //Reading the elements of the database
   for (i=0; i<N_ELEM; i++)
   {
//      printf("Reading vectors[%d] : ", i);
      for (j=0; j<dimension; j++)
      {
         fscanf(pf, "%f", &vectores[j][i]);
//         printf("%f ", vectores[i][j]);
      }
//      printf("\n");
      fgetc(pf);
   }
   fclose(pf);

   //Elems[dimension][N_ELEM]
   if (cudaSuccess != cudaMallocPitch((void **)&Elems, (size_t *)&pitch, N_ELEM*sizeof(float), (size_t)dimension))
      printf("\nERROR :: cudaMallocPitch 4\n");

   for (i=0; i < dimension; i++)
   {
      retorno = cudaMemcpy((float *)((char *)Elems + (i*(int)pitch)), (float *)(vectores[i]), sizeof(float)*N_ELEM, cudaMemcpyHostToDevice);
     if (retorno != cudaSuccess)
     {
      switch(retorno)
      {
       case cudaErrorInvalidPitchValue:
         printf("\nERROR 2 -> cudaErrorInvalidPitchValue:\n");
         break;
       case cudaErrorInvalidDevicePointer:
         printf("\nERROR 2 -> cudaErrorInvalidDevicePointer:\n");
         break;
       case cudaErrorInvalidMemcpyDirection:
         printf("\nERROR 2 -> cudaErrorInvalidMemcpyDirection:\n");
         break;
       case cudaErrorInvalidValue:
         printf("\nERROR 2 -> cudaErrorInvalidValue :: i=%d :: pitch=%d\n", i, pitch);
         break;
       default: 
         printf("\nERROR 2 -> Checkear esto.\n");
         break;
      }
      return 0;
     }
   }

   consultas =(float **)malloc(sizeof(float *)*N_QUERIES);
   for (i=0; i<N_QUERIES; i++)
      consultas[i] = (float *)malloc(sizeof(float)*dimension);

   if ((pf = fopen(argv[2], "r")) == NULL)
   {
      printf("\nNo se pudo abrir el archivo %s\n" ,argv[2]);
      return 0;
   }
/*
   fgets(linea, tam_lin-1, pf);
   fscanf(pf, "%d", &N_QUERIES);
   fscanf(pf, "%d", &dimension);
   fscanf(pf, "%d", &tam_elem);
   fgetc(pf);
   */
   printf("\n\nArchivo de Queries:\nCant. Elementos=%d :: dimension=%d\n" , N_QUERIES, dimension);

   //Reading the queries
   for (i=0; i<N_QUERIES; i++)
   {
	if (readdata(consultas[i], pf) == -1)
	{
		printf("\nError al leer Consultas\n");
         	cudaThreadExit();
		return 0;
	}
   }
   fclose(pf);

   //QUERY_dev[N_QUERIES][dimension]
   if (cudaSuccess != cudaMallocPitch((void **)&QUERY_dev, (size_t *)&pitch_Q, dimension*sizeof(float), (size_t)N_QUERIES))
      printf("\nERROR :: cudaMallocPitch 1\n");

   for (i=0; i < N_QUERIES; i++)
   {
     if (cudaSuccess != cudaMemcpy((char *)QUERY_dev + (i*(int)pitch_Q), consultas[i], sizeof(float)*dimension, cudaMemcpyHostToDevice))
       printf("\nERROR 3 :: cudaMemcpy\n");
   }

   //----------------------------
   T_per_BLOCK = N_ELEM;
   if (T_per_BLOCK > 512)
      T_per_BLOCK = 512;
   

   N_BLOQUES = Q;
   contQ = 0;
   cont = 0;
   getrusage(RUSAGE_SELF, &r1);
   gettimeofday(&t1, 0);

  while(contQ < N_QUERIES)
  {

      contQ += Q;
      if (contQ > N_QUERIES)
         N_BLOQUES = N_QUERIES - (contQ-Q);
      printf("\nN_BLOQUES = %d :: T_per_BLOCK = %d\n", N_BLOQUES, T_per_BLOCK);

      Batch_Heap_Reduction<<< N_BLOQUES, T_per_BLOCK>>> (Elems, (int)pitch, HEAPS_dev, (int)pitch_H, QUERY_dev, (int)pitch_Q, arr_Dist, (int)pitch_Dist, Q*cont, res_final);

      if (cudaSuccess != cudaMemcpy((float *)res_final_H, (float *)res_final, sizeof(float)*Q, cudaMemcpyDeviceToHost))
      {
         printf("\nERROR 41 :: cudaMemcpy :: iteraH\n");
         cudaThreadExit();
         return 0;
      }
      cont++;
  }

  gettimeofday(&t2, 0);


   real_time = (t2.tv_sec - t1.tv_sec) + (float)(t2.tv_usec - t1.tv_usec)/1000000;

   prom = 0;
   prom_cont = 0;
   for (i=0; i<Q; i++)
   {
         if (res_final_H[i] != 0)
         {
            prom += res_final_H[i];
            prom_cont += 1;
         }
   }

   printf("\nK = %d", TOPK);
   printf("\nReal Time = %f", real_time);
   printf("\nprom = %lf\n", (double)(prom/(double)prom_cont));
   fflush(stdout);

   cudaFree(Elems);
   cudaFree(QUERY_dev);
   cudaFree(HEAPS_dev);
   cudaFree(arr_Dist);

  cudaThreadExit();
  return 0;
}


__device__ void insertH(Elem *heap, Elem *elem, int *n_elem, int pitch, int id)
{
  int i;
  Elem temp;

  ((Elem *)((char *)heap + (*n_elem)*pitch))[id].dist = elem->dist;
  ((Elem *)((char *)heap + (*n_elem)*pitch))[id].ind = elem->ind;
  (*n_elem)++;
  for (i = *n_elem; i>1 && ((Elem *)((char *)heap + (i-1)*pitch))[id].dist > ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].dist; i=i/2)
  {
    //Intercambiamos con el padre
    temp.dist = ((Elem *)((char *)heap + (i-1)*pitch))[id].dist;
    temp.ind = ((Elem *)((char *)heap + (i-1)*pitch))[id].ind;
    ((Elem *)((char *)heap + (i-1)*pitch))[id].dist = ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].dist;
    ((Elem *)((char *)heap + (i-1)*pitch))[id].ind = ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].ind;
    ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].dist = temp.dist;
    ((Elem *)((char *)heap + ((i/2)-1)*pitch))[id].ind = temp.ind;
  }
  return;
}

__device__ void extractH(Elem *heap, int *n_elem, int pitch, int id, Elem *eresult)
{
  int i, k;
  Elem temp;
  eresult->dist = ((Elem *)((char *)heap+0))[id].dist; //Se guarda el maximo
  eresult->ind = ((Elem *)((char *)heap+0))[id].ind; 

  ((Elem *)((char *)heap+0))[id].dist = ((Elem *)((char *)heap + ((*n_elem)-1)*pitch))[id].dist;// Movemos el ultimo a la raiz y achicamos el heap
  ((Elem *)((char *)heap+0))[id].ind = ((Elem *)((char *)heap + ((*n_elem)-1)*pitch))[id].ind;
  (*n_elem)--;
  i = 1;
  while(2*i <= *n_elem) // mientras tenga algun hijo
  {
    k = 2*i; //el hijo izquierdo
    if(k+1 <= *n_elem && ((Elem *)((char *)heap + ((k+1)-1)*pitch))[id].dist > ((Elem *)((char *)heap + (k-1)*pitch))[id].dist)
      k = k+1;  //el hijo derecho es el mayor

    if(((Elem *)((char *)heap + (i-1)*pitch))[id].dist > ((Elem *)((char *)heap + (k-1)*pitch))[id].dist)
      break;  //es mayor que ambos hijos

    temp.dist = ((Elem *)((char *)heap + (i-1)*pitch))[id].dist;
    temp.ind = ((Elem *)((char *)heap + (i-1)*pitch))[id].ind;
    ((Elem *)((char *)heap + (i-1)*pitch))[id].dist = ((Elem *)((char *)heap + (k-1)*pitch))[id].dist;
    ((Elem *)((char *)heap + (i-1)*pitch))[id].ind = ((Elem *)((char *)heap + (k-1)*pitch))[id].ind;
    ((Elem *)((char *)heap + (k-1)*pitch))[id].dist = temp.dist;
    ((Elem *)((char *)heap + (k-1)*pitch))[id].ind = temp.ind;
    i = k;   //lo intercambiamos con el mayor hijo
  }
  return;
//  return max;
}


__device__ float topH(Elem *heap, int id)
{
  return ((Elem *)((char *)heap + 0))[id].dist;
}

__device__ void popush(Elem *heap, Elem *elem, int *n_elem, int pitch, int id)
{
  int i, k;
  Elem temp;

  ((Elem *)((char *)heap+0))[id].dist = elem->dist;
  ((Elem *)((char *)heap+0))[id].ind  = elem->ind;

  i = 1;
  while(2*i <= *n_elem) // mientras tenga algun hijo
  {
    k = 2*i; //el hijo izquierdo
    if(k+1 <= *n_elem && ((Elem *)((char *)heap + ((k+1)-1)*pitch))[id].dist > ((Elem *)((char *)heap + (k-1)*pitch))[id].dist)
      k = k+1;  //el hijo derecho es el mayor

    if(((Elem *)((char *)heap + (i-1)*pitch))[id].dist > ((Elem *)((char *)heap + (k-1)*pitch))[id].dist)
      break;  //es mayor que ambos hijos

    temp.dist = ((Elem *)((char *)heap + (i-1)*pitch))[id].dist;
    temp.ind = ((Elem *)((char *)heap + (i-1)*pitch))[id].ind;
    ((Elem *)((char *)heap + (i-1)*pitch))[id].dist = ((Elem *)((char *)heap + (k-1)*pitch))[id].dist;
    ((Elem *)((char *)heap + (i-1)*pitch))[id].ind = ((Elem *)((char *)heap + (k-1)*pitch))[id].ind;
    ((Elem *)((char *)heap + (k-1)*pitch))[id].dist = temp.dist;
    ((Elem *)((char *)heap + (k-1)*pitch))[id].ind = temp.ind;
    i = k;   //lo intercambiamos con el mayor hijo
  }
  return;
}


__global__ void Batch_Heap_Reduction(float *DB_dev, int pitch_DB, Elem *heap, int pitch_H, float *QUERY_dev, int pitch_QUERY, Elem *arr_Dist, int pitch_Dist, int beginQ, float *res_final)
{
  int i, j, n_elem=0, n_elemWarp=0;
  int id;
  Elem eresult;
  __shared__ Elem matrizWarp[TOPK][TAM_WARP];
  __shared__ Elem heapfin[TOPK][1];
  __shared__ float query[DIM];

  id = threadIdx.x + (blockDim.x * blockIdx.x);

  //Copying the query to shared memory
  for (i=threadIdx.x; i < DIM; i += blockDim.x)
      query[i] = ((float *)((char *)QUERY_dev + ((blockIdx.x + beginQ) * (int)pitch_QUERY)))[i];

  __syncthreads();

  //Getting the array of distances
  for (i=threadIdx.x; i < NE; i += blockDim.x)
  {
    ((Elem *)((char *)arr_Dist + (blockIdx.x*pitch_Dist)))[i].dist = distancia_trans(DB_dev, pitch_DB, i, query);
    ((Elem *)((char *)arr_Dist + (blockIdx.x*pitch_Dist)))[i].ind = i;
  }

  for(i=threadIdx.x; i < NE; i += blockDim.x)//NE = Numero de elementos de la BD
  {
      if (n_elem >= TOPK)
      {
         if (topH(heap, id) > ((Elem *)((char *)arr_Dist + (blockIdx.x*pitch_Dist)))[i].dist)
            popush(heap, &(((Elem *)((char *)arr_Dist + (blockIdx.x*pitch_Dist)))[i]), &n_elem, pitch_H, id); //pop and push in a single operation
      }
      else
         insertH(heap, &(((Elem *)((char *)arr_Dist + (blockIdx.x*pitch_Dist)))[i]), &n_elem, pitch_H, id);
  }

  __syncthreads();


  //The first warp of the CUDA Block reduces the problem to a matrix of size Kx32 elements, but storing the elements of the heaps in shared memory
  if (threadIdx.x < TAM_WARP)
  {
    for(j=id; j < blockDim.x*(blockIdx.x+1); j += TAM_WARP)
    {
       n_elem = TOPK;
       for(i=0; i < TOPK; i++)
       {
         extractH(heap, &n_elem, pitch_H, j, &eresult);

         if (n_elemWarp < TOPK)
           insertH(&(matrizWarp[0][0]), &eresult, &n_elemWarp, sizeof(Elem)*TAM_WARP, threadIdx.x);
         else
           if (topH(&(matrizWarp[0][0]), threadIdx.x) > eresult.dist)
             popush(&(matrizWarp[0][0]), &eresult, &n_elemWarp, sizeof(Elem)*TAM_WARP, threadIdx.x);
       }
    }
  }
  

  __syncthreads();


  //the first thread of the CUDA Block finds the k results from the previous matrix of size TOPKxTAM_WARP
  if (threadIdx.x == 0)
  {
     n_elem = 0;
     for(j=0; j < TAM_WARP; j++)
     {
       for(i=0; i < TOPK; i++)
          if (n_elem < TOPK)
             insertH((Elem *)heapfin, &(matrizWarp[i][j]), &n_elem, sizeof(Elem), 0);
          else
             if (topH((Elem *)heapfin, 0) > matrizWarp[i][j].dist)
               popush((Elem *)heapfin, &(matrizWarp[i][j]), &n_elem, sizeof(Elem), 0);
     }

     //Writing the closest element to the query
     res_final[blockIdx.x] = topH((Elem *)heapfin, 0);

     //To write te TOPK elements retrieved in this function, you must use an array 'arr_res1' with size TOPK*Q, and to do as follows:
//     for (i=TOPK*blockIdx.x; i<TOPK*(blockIdx.x+1); i++)
//        extractH(&(heapfin[0][0]), &n_elem, sizeof(Elem), 0, &(arr_res1[i]));
  }

  return;
}

__device__ float distancia_trans(float *p1, int pitch_p1, int col_1, float *q)
{
   int i=0;
   float suma=0;

   for (i=0; i < DIM; i++)
      suma += (((float *)((char *)p1 + (i*pitch_p1)))[col_1] - q[i]) * 
              (((float *)((char *)p1 + (i*pitch_p1)))[col_1] - q[i]);

   return sqrtf(suma);
}


int readdata(float *dato, FILE *file)
{
   int i=0;
   
   for (i=0;i<DIM;i++)
      if (fscanf(file,"%f",&dato[i])<1)
         return -1;
   return 1;
}
