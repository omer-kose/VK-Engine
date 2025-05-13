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

- A proper scene structure to store scene related resources such as loaded scenes, meshes, camera, lights and more
- Even though passes are abstracted out, drawing geometry is still done by calling them in drawGeometry function inside vk_engine. Add an abstraction over VulkanEngine to be able to write samples without touching anything in the engine.
- Add actual PBR shading. Currently, I have a placeholder 
- Add Deferred Rendering Support (Adding G and Light Passes more accurately)
- Add OBJ file loading
- and many more (implementing Graphics techniques inside the engine)

## Screenshots

Structure Scene (PBR Shading is not yet implemented)
![image](https://github.com/user-attachments/assets/88583114-87c6-4939-9380-0d9246d9ecc8)

## References
Mainly built upon the final version of: https://vkguide.dev/

