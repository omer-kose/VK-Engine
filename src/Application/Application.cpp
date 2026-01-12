#include "Application.h"

#include <SDL.h>

void SK::Application::init(Application* application, uint32_t windowWidth, uint32_t windowHeight)
{
    assert(application->isInitialized == false);

    application->windowWidth = windowWidth;
    application->windowHeight = windowHeight;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    application->window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        application->windowWidth,
        application->windowHeight,
        window_flags
    );

    application->isInitialized = true;
}
