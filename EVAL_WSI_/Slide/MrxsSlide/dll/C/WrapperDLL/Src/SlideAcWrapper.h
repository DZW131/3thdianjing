#ifndef __SLIDE_AC_WRAPPER_H
#define __SLIDE_AC_WRAPPER_H
#include "SlideAC.h"
// Open Slide
EXTERN_C __declspec(dllexport) ITDHSlideAC* WINAPI OpenSlide(LPCSTR SlideName);
// Close slide
EXTERN_C __declspec(dllexport) BOOL WINAPI CloseSlide(ITDHSlideAC* SlideAC);
EXTERN_C __declspec(dllexport) int WINAPI GetIntSlideProperties(ITDHSlideAC* SlideAC, int flag);
EXTERN_C __declspec(dllexport) double WINAPI GetDoubleSlideProperties(ITDHSlideAC* SlideAC, int flag);
EXTERN_C __declspec(dllexport) int WINAPI GetImage(ITDHSlideAC* SlideAC, long X1, long Y1, long X2, long Y2, long Magnify, unsigned char * point);
EXTERN_C __declspec(dllexport) int WINAPI GetAssociatedImages(ITDHSlideAC* SlideAC, long ImageType, unsigned char * point);
//EXTERN_C __declspec(dllexport) int WINAPI GetAssociatedImagesProperties(ITDHSlideAC* SlideAC, long ImageType, long flag);
EXTERN_C __declspec(dllexport) BOOL WINAPI GetAssociatedImagesProperties(ITDHSlideAC* SlideAC, long ImageType, int* s, int* width, int* height);

// Readout of scan map into a standard Windows bitmap
//EXTERN_C __declspec(dllexport) BOOL WINAPI GetScanmap(long Magnify, LPBITMAPINFOHEADER* Scanmap);
// Returns an image with 256 x 256 pixel resolution from tile coordinate X, Y into a standard Windows bitmap
//EXTERN_C __declspec(dllexport) BOOL WINAPI GetImage(long X, long Y, long Magnify, LPBITMAPINFOHEADER* Image);

#endif //__SLIDE_AC_WRAPPER_H