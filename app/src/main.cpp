#include "framework.h"
#include "windows_handler.h"

#define MAX_LOADSTRING 100

const wchar_t windowClassName[] = L"CUDA_SAMPLE";

#define WM_CUDA_SAMPLE_MSG_LOAD_IMAGE   (WM_USER + 1)

typedef struct {
    Bitmap_t* bitmap;
    Windows_t* windows;
	cuda_gfx_t* cudaGFX;
} AppWindowData;


LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam){

    AppWindowData* data = reinterpret_cast<AppWindowData*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

    if (data) {
        Bitmap_t* bmp = data->bitmap;
        Windows_t* win = data->windows;
    }

    switch (message) {
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc;
            hdc = ::BeginPaint(data->windows->windowsHandle, &ps);

            if (data->windows->windowsHandle && data->bitmap) {
                RECT clientRect;
                GetClientRect(data->windows->windowsHandle, &clientRect);
                int clientWidth = clientRect.right - clientRect.left;
                int clientHeight = clientRect.bottom - clientRect.top;

                int x = (clientWidth - data->bitmap->width) / 2;
                int y = (clientHeight - data->bitmap->height) / 2;

                drawBitmap(data->windows->windowsHandle, data->bitmap, x, y);
            }

            ::EndPaint(data->windows->windowsHandle, &ps);
            break;
        }

        case WM_DESTROY: {
            PostQuitMessage(0);
            break;
        }

        case WM_CUDA_SAMPLE_MSG_DESTROY: {
            Windows_t* pWin = reinterpret_cast<Windows_t*>(lParam);
            if (pWin) {
                pWin->isRunning = false;
                windowsDestroy(pWin);
            }
            break;
        }

        case WM_KEYDOWN: {
            if (wParam == VK_ESCAPE) {
                PostQuitMessage(0);
            }
            break;
        }

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}



int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow) {

    Windows_t win = {0};
	wcscpy_s(win.className, L"CUDA_SAMPLE");
	windows_create(&win, hInstance, WndProc);

    Bitmap_t bitmap = { 0 };
    windowsBitmapCreate(win.windowsHandle, &bitmap, 256, 256, 3);

	cuda_gfx_t cudaGFX = { 0 };

	double to_device_copy_time = toDevice(&cudaGFX, bitmap.data, bitmap.width, bitmap.height);
	double execution_time = run(&cudaGFX);
	double to_host_copy_time =  toHost(&cudaGFX, bitmap.data);


    AppWindowData* data = new AppWindowData{ &bitmap, &win, &cudaGFX };
    // Before calling SetWindowLongPtr, check that win.windowsHandle is not NULL (0)
    if (win.windowsHandle != NULL && win.windowsHandle != 0) {
        SetWindowLongPtr(win.windowsHandle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(data));
    }

    while (win.isRunning) {
		windowsGetEvents(&win);
    }

    windowsDestroy(&win);
	destroyBitmap(&bitmap);
	cleanup(&cudaGFX);

    return 1;
}