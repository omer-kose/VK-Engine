#pragma once

#include <stdint.h>

// Forward declare in the global scope to make SDL_Window namespace independent. 
struct SDL_Window;

namespace SK::Application
{
	struct Application
	{
		SDL_Window* window;
		uint32_t windowWidth = 1920;
		uint32_t windowHeight = 1080;
		bool isInitialized = false;
	};

	void init(Application* app, uint32_t windowWidth = 1920, uint32_t windowHeight = 1080);
};
