# VK Engine

## Features
- Vulkan 1.3 (with VK Bootstrap and VMA)
- Frames in flight
- Dynamic Rendering
- ImGui
- GLTF Scene Loader
- Abstraction for Materials and Passes

## TODO List
I develop the engine continuously.

- Compile times are disgustingly long whenever a header file is modified. Implement Interface/Implementation split or PIMPL for the Engine class.
- Setup precompiled headers for the external libraries to boost compilation times again.
- Realized that static pass idea is simple but too restrictive for multiple reasons. Refactor it into something more dynamic and flexible (most likely into Renderers that makes use of passes).
- Implement Render or Frame Graphs.
- Introduce scenes, which I have already done in the thesis code.
- Refactor scene and mesh loading as well as the material system.
- Add actual PBR shading. Currently, I have a placeholder 
- Add Deferred Rendering Support (Adding G and Light Passes more accurately)
- Add OBJ file loading
- and many more (implementing Graphics techniques inside the engine)

## Screenshots

Structure Scene (PBR Shading is not yet implemented)
![image](https://github.com/user-attachments/assets/88583114-87c6-4939-9380-0d9246d9ecc8)

## References
Mainly built upon the final version of: https://vkguide.dev/





