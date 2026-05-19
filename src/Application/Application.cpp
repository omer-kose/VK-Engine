#include "Application.h"

#include <SDL.h>

#include <UI/UI.h>

void SK::Application::init(State* application, uint32_t windowWidth, uint32_t windowHeight)
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

void SK::Application::handleSDLEvents(State* application, SDLEventCallback eventCallback, void* eventContext)
{
    SDL_Event e;

    // Handle events on queue
    while(SDL_PollEvent(&e) != 0)
    {
        // close the window when user alt-f4s or clicks the X button
        if(e.type == SDL_QUIT)
        {
            application->shouldQuit = true;
        }

        if(e.type == SDL_WINDOWEVENT)
        {
            if(e.window.event == SDL_WINDOWEVENT_MINIMIZED)
            {
                application->isMinimized = true;
            }
            if(e.window.event == SDL_WINDOWEVENT_RESTORED)
            {
                application->isMinimized = false;
            }
        }

        if (eventCallback != nullptr)
        {
            eventCallback(e, eventContext);
        }
    }
}

void SK::Application::shutdown(State* application)
{
    if(application->isInitialized)
    {
        SDL_DestroyWindow(application->window);
        application->window = nullptr;
        application->isInitialized = false;
    }

    application->isInitialized = false;
}
