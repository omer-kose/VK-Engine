# VK Engine

## Features
- Vulkan 1.3 (with VK Bootstrap and VMA)
- Frames in flight
- Dynamic Rendering
- ImGui
- GLTF Scene Loader
- Abstraction for Materials and Passes

## TODO List
Besides developing a fully-fledged graphics engine (or framework), I also use this project to improve my large software architecture skills. So, the code is subject to big changes or rewrites. 

- Convert the engine into an importable backend library independent of the application which will use it.
- Setup precompiled headers for the external libraries to boost compilation times.
- Realized that static pass class idea is simple but too restrictive for multiple reasons. Refactor it into something more dynamic and flexible.
- Add SLANG support.
- Add a proper shader compilation system.
- Propose a proper material system.
- Add Deferred Rendering Support (Adding G and Light Passes more accurately).
- Add actual PBR shading. Currently, I have a placeholder.
- Add OBJ file loading.
- Implement Render or Frame Graphs.
- and many more (implementing Graphics techniques using the engine as the framework).

## Screenshots

Structure Scene (PBR Shading is not yet implemented)
![image](https://github.com/user-attachments/assets/88583114-87c6-4939-9380-0d9246d9ecc8)

## References
Built upon the final version of: https://vkguide.dev/















