#pragma once

#include <stdint.h>

// Unless I find a better place, camera will be in Application Layer
#include <camera.h>

// Forward declare SDL_Window in the global scope to make it independent of the namespace. 
// SDL.h is not included in the header file as I don't want to expose SDL implementation and its types outside of the Application (Platform) layer.
struct SDL_Window;

namespace SK::Application
{
	struct Application
	{
		SDL_Window* window;
		uint32_t windowWidth = 1920;
		uint32_t windowHeight = 1080;
		
		Camera mainCamera;

		bool isMinimized = false;
		bool shouldQuit = false;

		bool isInitialized = false;
	};

	void init(Application* application, uint32_t windowWidth = 1920, uint32_t windowHeight = 1080);
	void handleSDLEvents(Application* application);
	void initCamera(Application* application, glm::vec3 position, float pitch, float yaw);

	void shutdown(Application* application);
};