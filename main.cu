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
    
    hit_record rec;
    while (depth > 0 && (*world)->hit(r, 0.001, MAXFLOAT, rec)) {
        ray scattered;
        vec3 attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, random_state)) {
            res *= attenuation;
            r = scattered;
            --depth;
        } else {
            return vec3(0,0,0);
        }
        --depth;
    }
    
    if(depth <= 0){
        return vec3(0,0,0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5*(unit_direction.y() + 1.0);
    res *= (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);

    return res;
}

__global__ void createWorldKernel(hitable ** d_world, hitable ** d_list, camera ** d_cam,int nx, int ny, int n,curandState *state){
    if(blockIdx.x == 0 && threadIdx.x == 0){
        int grid = (int)sqrt(1.0*n); 

        d_list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a =  - grid / 2; a <  grid /2; a++) {
            for (int b = - grid / 2; b < grid / 2; b++) {
                float choose_mat = curand_uniform(state);
                vec3 center(a+0.9*curand_uniform(state),0.2,b+0.9*curand_uniform(state));
                if ((center-vec3(4,0.2,0)).length() > 0.9) {
                    if (choose_mat < 0.8) {  // diffuse
                        d_list[i++] = new sphere(center, 0.2,
                            new lambertian(vec3(curand_uniform(state)*curand_uniform(state),
                                                curand_uniform(state)*curand_uniform(state),
                                                curand_uniform(state)*curand_uniform(state))
                            )
                        );
                    }
                    else if (choose_mat < 0.95) { // metal
                        d_list[i++] = new sphere(center, 0.2,
                                new metal(vec3(0.5*(1 + curand_uniform(state)),
                                            0.5*(1 + curand_uniform(state)),
                                            0.5*(1 + curand_uniform(state))), 
                                        0.5*curand_uniform(state)));
                    }
                    else {  // glass
                        d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                    }
                }
            }
        }

        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

        *d_world   = new hitable_list(d_list,n);

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
        int pixel_index = j*nx + i;

        for (int s=0; s < ns; s++) {
            float u = float(i + curand_uniform(&pixal_states[pixel_index])) / float(nx);
            float v = float(j + curand_uniform(&pixal_states[pixel_index])) / float(ny);
            ray r = (*cam)->get_ray(u, v);
            
            col += color(r, d_world, max_depth,&pixal_states[pixel_index]);
        }
        col /= float(ns);
        col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );
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
    int nx = 1000;
    int ny = 500;
    int ns = 10;
    int max_depth = 50;
    int no_object = 100;
    vec3 *fbuffer;
    curandState *pixal_states;

    CHECK(cudaMallocManaged(&fbuffer, nx * ny * sizeof(vec3)));
    
    
    hitable **d_world;
    hitable **d_list;
    camera ** d_cam;
    CHECK(cudaMalloc((void **)&pixal_states, nx*ny *sizeof(curandState)));
    CHECK(cudaMalloc((void **)&d_list, no_object*sizeof(hitable *)));
    CHECK(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    CHECK(cudaMalloc((void **)&d_cam, sizeof(camera *)));

    dim3 blockSize(8,8);
    dim3 gridSize((nx -1) /blockSize.x + 1, (ny - 1)/ blockSize.y + 1);

    render_init<<<gridSize, blockSize >>>(nx, ny, pixal_states);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    createWorldKernel<<<1,1>>>(d_world, d_list, d_cam, nx , ny, no_object, &pixal_states[0]);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    renderKernel<<<gridSize,blockSize>>>(fbuffer, d_world, d_cam, nx, ny, ns, max_depth, pixal_states);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            vec3 col = fbuffer[j * nx + i];
            int ir = 255.99 * col.r();
            int ig = 255.99 * col.g();
            int ib = 255.99 * col.b();
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    cudaFree(fbuffer);

    return 0;
}


