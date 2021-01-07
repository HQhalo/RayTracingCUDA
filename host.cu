#include <iostream>
#include "shape/sphere.h"
#include "shape/hitable_list.h"
#include <float.h>
#include "camera/camera.h"
#include "material/metal.h"
#include "material/dielectric.h"
#include "material/lambertian.h"

// vec3 color(const ray& r, hitable *world, int depth) {
//     hit_record rec;
//     if(depth <= 0) {
//         return vec3(0,0,0);
//     }
//     if (world->hit(r, 0.001, MAXFLOAT, rec)) {
//         ray scattered;
//         vec3 attenuation;
//         if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
//             return attenuation*color(scattered, world, depth - 1);
//         }
//         return vec3(0,0,0);
//     }
//     vec3 unit_direction = unit_vector(r.direction());
//     float t = 0.5*(unit_direction.y() + 1.0);
//     return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
// }

vec3 color(ray r, hitable *world, int depth) {
    vec3 res = vec3(1, 1, 1);
    
    vec3* attenuationCache = new vec3[depth];
    size_t attenuationCacheSize = 0;

    hit_record rec;
    while (depth > 0 && world->hit(r, 0.001, MAXFLOAT, rec)) {
        ray scattered;
        vec3 attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
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

hitable *random_scene() {
    int n = 100;
    int grid = (int)sqrt(n); 
    hitable **list = new hitable*[n+1];
    list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a =  - grid / 2; a <  grid /2; a++) {
        for (int b = - grid / 2; b < grid / 2; b++) {
            float choose_mat = drand48();
            vec3 center(a+0.9*drand48(),0.2,b+0.9*drand48());
            if ((center-vec3(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new sphere(center, 0.2,
                        new lambertian(vec3(drand48()*drand48(),
                                            drand48()*drand48(),
                                            drand48()*drand48())
                        )
                    );
                }
                else if (choose_mat < 0.95) { // metal
                    list[i++] = new sphere(center, 0.2,
                            new metal(vec3(0.5*(1 + drand48()),
                                           0.5*(1 + drand48()),
                                           0.5*(1 + drand48())), 
                                      0.5*drand48()));
                }
                else {  // glass
                    list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
    }

    // list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
    // list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

    return new hitable_list(list,i);
}

int main() {
    int nx = 1000;
    int ny = 500;
    int ns = 10;
    int max_depth = 50;
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    
    hitable *world = random_scene();

    vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    vec3 vup(0,1,0);
    camera cam(lookfrom, lookat, vup, 20, float(nx)/float(ny));

    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            vec3 col(0, 0, 0);
            for (int s=0; s < ns; s++) {
                float u = float(i + drand48()) / float(nx);
                float v = float(j + drand48()) / float(ny);
                ray r = cam.get_ray(u, v);
                col += color(r, world, max_depth);
            }
            col /= float(ns);
            col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );
            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
}