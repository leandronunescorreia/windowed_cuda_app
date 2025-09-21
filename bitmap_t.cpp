#include "bitmap_t.h"


long windowsBitmapCreate(HWND windowsHandle, Bitmap_t* self, uint32_t width, uint32_t height, uint8_t bitDepth) {
    BITMAPINFO bmpInfo;
    bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmpInfo.bmiHeader.biWidth = width;
    bmpInfo.bmiHeader.biHeight = -((LONG)height); // Negative height for top-down bitmap
    bmpInfo.bmiHeader.biPlanes = 1;
    bmpInfo.bmiHeader.biBitCount = bitDepth * 8; // bits per pixel
    bmpInfo.bmiHeader.biCompression = BI_RGB;
    bmpInfo.bmiHeader.biSizeImage = 0;
    bmpInfo.bmiHeader.biXPelsPerMeter = 0;
    bmpInfo.bmiHeader.biYPelsPerMeter = 0;
    bmpInfo.bmiHeader.biClrUsed = 0;
    bmpInfo.bmiHeader.biClrImportant = 0;

    HBITMAP hBitmap = CreateDIBSection(0, &bmpInfo, DIB_RGB_COLORS, (void**)&self->data, NULL, 0);

    if (hBitmap) {
        self->hBitmap = hBitmap;
        self->width = width;
        self->height = height;
        self->bitDepth = bitDepth;
        self->lineStride = calcLineStride(self);
        return 0; // Success
    }
    else {
        return -1; // Failure
    }

}

void drawBitmap(HWND handle, Bitmap_t* btmp, int32_t x, int32_t y) {
    HDC hdc = GetDC(handle);
    HDC hMemDC = CreateCompatibleDC(hdc);

    HGDIOBJ oldBitmap = SelectObject(hMemDC, btmp->hBitmap);
    BitBlt(hdc, x, y, btmp->width, btmp->height, hMemDC, 0, 0, SRCCOPY);
    SelectObject(hMemDC, oldBitmap);
    DeleteDC(hMemDC);
}

void destroyBitmap(Bitmap_t* btmp) {
    if (btmp->hBitmap) {
        DeleteObject(btmp->hBitmap);
        btmp->hBitmap = NULL;
        btmp->data = NULL; // The memory is freed by DeleteObject
    }
}

size_t calcLineStride(Bitmap_t* btmp) {
    return btmp->bitDepth * btmp->width;
}

void sexPixel(Bitmap_t* btmp, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    const size_t pixelStride = x * btmp->bitDepth;
    const size_t index = (y * btmp->lineStride) + pixelStride;
    btmp->data[index] = b;
    btmp->data[index + 1] = g;
    btmp->data[index + 2] = r;
}

uint8_t* gexPixel(Bitmap_t* btmp, uint32_t x, uint32_t y, uint8_t* r, uint8_t* g, uint8_t* b) {
    const size_t pixelStride = x * btmp->bitDepth;
    const size_t index = (y * btmp->lineStride) + pixelStride;

    uint8_t* rgb = new uint8_t[3];
    rgb[0] = btmp->data[index];
    rgb[1] = btmp->data[index + 1];
    rgb[2] = btmp->data[index + 2];
    return rgb;
}