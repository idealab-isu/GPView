# GPView

## Authors

* Dr. Adarsh Krishnamurthy *(adarsh@iastate.edu)*
* Onur R. Bingol *(orbingol@iastate.edu)*
* Sambit Ghadai *(sambitg@iastate.edu)*
* Aditya Balu *(baditya@iastate.edu)*
* Gavin Young
* Xin Huang

## Introduction

**GPU Accelerated Voxelization Framework** of 3D CAD models.

GPView voxelizes CAD models with a hybrid resolution. Supported CAD models are Wavefront **(.obj)** file and **.off** file formats. Output is in the form of binary **.raw** files with contiguous array.

Level1 resolution is a coarse level of voxelization and Level2 is a finer level of voxelization of particular Level1 voxels in a nested structure.

For more details visit [IDEALab](http://web.me.iastate.edu/idealab/index.html) website.

### Publications

1. Sambit Ghadai, Aditya Balu, Soumik Sarkar, Adarsh Krishnamurthy; [Learning localized features in 3D CAD models for manufacturability analysis of drilled holes](https://www.sciencedirect.com/science/article/pii/S0167839618300384), Computer Aided Geometric Design, 62:263-275, 2018.
2. Gavin Young, Adarsh Krishnamurthy; [GPU-accelerated generation and rendering of multi-level voxel representations of solid models](https://www.sciencedirect.com/science/article/pii/S009784931830102X), Computers & Graphics, 75:11-24, 2018.
3. Sambit Ghadai, Xian Lee, Aditya Balu, Soumik Sarkar, Adarsh Krishnamurthy; [Multi-Resolution 3D Convolutional Neural Networks for Object Recognition](https://arxiv.org/abs/1805.12254), NVIDIA GPU Technology Conference, arXiv:1805.12254, 2018.
4. Sambit Ghadai, Aditya Balu, Adarsh Krishnamurthy, Soumik Sarkar; [Learning and visualizing localized geometric features using 3D-CNN: An application to manufacturability analysis of drilled holes](https://arxiv.org/abs/1711.04851), NIPS Symposium on Interpretable Machine Learning, 2017.

## How to run GPView

### Minimum Requirements

* [GLEW v1.13.0](http://glew.sourceforge.net/) libraries
* [Freeglut v2.8.1](https://www.transmissionzero.co.uk/software/freeglut-devel/) libraries
* [CMake v3.7](https://cmake.org/download/)
* [NVIDIA CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-toolkit) along with [NVIDIA CUDA GPU](https://developer.nvidia.com/cuda-gpus)s for GPU acceleration.
* Microsoft Visual Studio 12 2013 for Windows compilation.
* GCC v5.4.0 for Linux compilation.

### Setup

Setup instructions for Windows and Linux can be found on wiki page.

## Guidelines

Guidelines to voxelize CAD models using GPView can be found on the wiki page.

## License

[MIT](LICENSE)

## Acknowledgements

### Additional Code References

#### TriRayIntersection.cpp

```
Triangle ray intersection test routine,
Tomas Muller and Ben Trumbore, 1997.
See article "Fast, Minimum Storage Ray/Triangle Intersection,"
Muller & Trumbore. Journal of Graphics Tools, 1997.
```

#### TriBoxIntersection.cpp

```
AABB-triangle overlap test code                     
by Tomas Akenine-Muller                             
Function: int triBoxOverlap(float boxcenter[3],     
          float boxhalfsize[3],float triverts[3][3]);
```
