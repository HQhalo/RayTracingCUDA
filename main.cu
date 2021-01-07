#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include "shape/hitable.h"
#include "shape/hitable_list.h"
#include "shape/sphere.h"
#include "common/vec3.h"
#include "camera/camera.h"
#include "material/metal.h"
#include "material/dielectric.h"
#include "material/lambertian.h"
#include <curand_kernel.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

__device__  vec3 color(ray r, hitable **world, int depth, curandState *random_state) {
    vec3 res = vec3(1, 1, 1);
    
    vec3* attenuationCache = new vec3[depth];
    size_t attenuationCacheSize = 0;

    hit_record rec;
    while (depth > 0 && (*world)->hit(r, 0.001, MAXFLOAT, rec)) {
        ray scattered;
        vec3 attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, random_state)) {
            attenuationCache[attenuationCacheSize++] = attenuation;
            r = scattered;
            --depth;
        } else {
            return vec3(0,0,0);
        }
    }

    if(depth <= 0){
        return vec3(0,0,0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5*(unit_direction.y() + 1.0);
    res = (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);

    for (int i = attenuationCacheSize - 1; i >= 0; --i) {
        res *= attenuationCache[i];
    }

    delete[]attenuationCache;

    return res;
}

__global__ void createWorldKernel(hitable ** d_world, hitable ** d_list, camera ** d_cam,curandState * pixal_states,int nx, int ny){
    if(blockIdx.x == 0 && threadIdx.x == 0){
        d_list[0] = new sphere(vec3(0,0,-1), 0.5,
                               new lambertian(vec3(0.8, 0.3, 0.3)));
        d_list[1] = new sphere(vec3(0,-100.5,-1), 100,
                               new lambertian(vec3(0.8, 0.8, 0.0)));
        *d_world   = new hitable_list(d_list,2);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        vec3 vup(0,1,0);

        *d_cam = new camera(lookfrom, lookat, vup, 20, float(nx)/float(ny));
    }
}

__global__ void renderKernel(vec3 *fbuffer,hitable ** d_world, camera **cam, int nx, int ny, int ns, int max_depth, curandState * pixal_states){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
   
    if(i < nx && j < ny){
        vec3 col(0, 0, 0);
        for (int s=0; s < ns; s++) {
            float u = float(i + curand_uniform(&pixal_states[j* nx + i])) / float(nx);
            float v = float(j + curand_uniform(&pixal_states[j* nx + i])) / float(ny);
            ray r = (*cam)->get_ray(u, v);
            col += color(r, d_world, max_depth,&pixal_states[j* nx + i]);
        }
        col /= float(ns);
        col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );
        printf("a%f\n",col[0]);
        fbuffer[j * nx + i] = col;
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return;
   int pixel_index = j*max_x + i;
   curand_init(42, pixel_index, 0, &rand_state[pixel_index]);
}


int main(){
    int nx = 500;
    int ny = 250;
    int ns = 10;
    int max_depth = 50;
    vec3 *fbuffer;
    curandState *pixal_states;

    CHECK(cudaMallocManaged(&fbuffer, nx * ny * sizeof(vec3)));
    
    
    hitable **d_world;
    hitable **d_list;
    camera ** d_cam;
    CHECK(cudaMalloc((void **)&pixal_states, nx*ny *sizeof(curandState)));
    CHECK(cudaMalloc((void **)&d_list, 4*sizeof(hitable *)));
    CHECK(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    CHECK(cudaMalloc((void **)&d_cam, sizeof(camera *)));

    dim3 blockSize(8,8);
    dim3 gridSize((nx -1) /blockSize.x + 1, (ny - 1)/ blockSize.y + 1);

    render_init<<<gridSize, blockSize >>>(nx, ny, pixal_states);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    createWorldKernel<<<1,1>>>(d_world, d_list, d_cam, pixal_states, nx , ny);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    renderKernel<<<gridSize,blockSize>>>(fbuffer, d_world, d_cam, nx, ny, ns, max_depth, pixal_states);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    // std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    // for (int j = ny-1; j >= 0; j--) {
    //     for (int i = 0; i < nx; i++) {
    //         vec3 col = fbuffer[j * nx + i];
    //         int ir = 255.99 * col.r();
    //         int ig = 255.99 * col.g();
    //         int ib = 255.99 * col.b();
    //         std::cout << ir << " " << ig << " " << ib << "\n";
    //     }
    // }
    cudaFree(fbuffer);

    return 0;
}


