#include "windows_handler.h"



long windows_create(Windows_t* parent, HINSTANCE hInstance, WNDPROC windowsProc) {
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = windowsProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    wcex.hIcon = nullptr;
    wcex.lpszMenuName = nullptr;
    wcex.hIconSm = nullptr;

    wcex.lpszClassName = parent->className;


    if (!RegisterClassExW(&wcex)) {
        return 0;
    }


    HWND hWnd = CreateWindowW(
        wcex.lpszClassName,
        wcex.lpszClassName,
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        nullptr,
        nullptr,
        hInstance,
        nullptr);

    if (!hWnd) {
        return 0;
    }

    parent->windowsHandle = hWnd;
    parent->isRunning = true;

    return 1;
}

void windowsDestroy(Windows_t* parent) {
    if (parent->windowsHandle) {
        DestroyWindow(parent->windowsHandle);
        parent->windowsHandle = nullptr;
        parent->isRunning = false;
    }
    parent->isRunning = false;
}

long windowsGetEvents(Windows_t* parent) {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            parent->isRunning = false;
            PostMessage(parent->windowsHandle, WM_CUDA_SAMPLE_MSG_DESTROY, 0, reinterpret_cast<LPARAM>(parent));
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 1;
}