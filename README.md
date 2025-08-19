# VK Engine

## Features
- Vulkan 1.3 (with VK Bootstrap and VMA)
- Frames in flight
- Dynamic Rendering
- ImGui
- GLTF Scene Loader
- Abstraction for Materials and Passes

## TODO List
I develop the engine continuously. (Currently, I am working on my Master's thesis)

- With the arrival of Vulkan 1.4, and me exploring more flexible stuff like Push Descriptors, I will refactor the engine. Most probably, I will go for a simpler and more flexible Pass approach by implementing a simple framegraph. While working on my thesis, I realized that working with static passes can be a bit cumbersome. Also, it blocks multi-threaded rendering possibilities. So, I plan a much bigger refactor by analyzing other architectures.
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


