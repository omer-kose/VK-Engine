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

- The engine is not really designed around OOP but I might totally get rid of it as I prefer (and actually learn more about) group based thinking rather than element based thinking.
- As I have seen in the thesis implementation, compile times tend to get disgustingly long whenever a header file is modified. Fix it.
- Setup precompiled headers for the external libraries to boost compilation times further.
- Realized that static pass class idea is simple but too restrictive for multiple reasons. Refactor it into something more dynamic and flexible.
- Convert the engine into an importable backend library independent of the application which will use it.
- Design and implement the renderer frontend that will use the backend.
- Move to SLANG from GLSL.
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












