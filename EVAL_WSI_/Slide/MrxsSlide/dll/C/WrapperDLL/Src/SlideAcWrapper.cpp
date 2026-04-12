#include <windows.h>
#include <atlbase.h>
#include "SlideAC.h"
#include "SlideAC_i.c"
#include "SlideAcWrapper.h"
#include <cstdlib>
#include <vector>
#include <mutex>

using namespace std;

static BOOL ConvertTDHToStdBitmap(LPBITMAPINFOHEADER* Bitmap);
static std::vector<CComPtr<ITDHSlideAC>> list(0);





EXTERN_C BOOL WINAPI DllMain(HINSTANCE hInstance, DWORD dwReason, LPVOID /*lpReserved*/)
{
	if (dwReason == DLL_PROCESS_ATTACH) {
		//printf("dllMain中初始化了");
		DisableThreadLibraryCalls(hInstance);
		::CoInitialize(NULL);
	}
	else if (dwReason == DLL_PROCESS_DETACH) {
		auto ite = list.begin();
		while (ite != list.end()) {
			//ite->Release();
			//printf("dllMain中关闭了%p", *ite);
			ite = list.erase(ite);
			}
		::CoUninitialize();
	}
	return TRUE;
}

EXTERN_C __declspec(dllexport) ITDHSlideAC* WINAPI OpenSlide(LPCSTR SlideName)
{
	HRESULT hr;
	::CoInitialize(NULL);
	CComPtr<ITDHSlideAC> SlideAC;
	if ((hr = SlideAC.CoCreateInstance(CLSID_TDHSlideAC)) == S_OK)
	{
		CComBSTR bstr(SlideName);

		if ((hr = SlideAC->OpenSlide(bstr)) == S_OK)
		{
			list.push_back(SlideAC);
			return SlideAC;
		}
	}

	return NULL;
}

EXTERN_C __declspec(dllexport) BOOL WINAPI CloseSlide(ITDHSlideAC* SlideAC)
{

	//printf("dll中的CloseSlide在auto前关闭了%p", ac);
	auto ite = list.begin();
	while (ite != list.end()) {
		if (*ite == SlideAC) {
			//printf("dll中的CloseSlide在Release()前关闭了%p", *ite);
			//SlideAC->Release();

			list.erase(ite);
			return true;
		}
		else {
			ite++;
		}
	}

	return false;
}

EXTERN_C __declspec(dllexport) int WINAPI GetIntSlideProperties(ITDHSlideAC* SlideAC, int flag) {
	HRESULT hr;
	VARIANT res;
	res.vt = 19;
	VariantInit(&res);

	if ((hr = SlideAC->GetSlideProperties(flag, &res)) == S_OK)
	{
		return res.intVal;
	}

	return 0;
}


EXTERN_C __declspec(dllexport)	double WINAPI GetDoubleSlideProperties(ITDHSlideAC* SlideAC, int flag) {
	HRESULT hr;
	VARIANT res;
	VariantInit(&res);
	res.vt = VT_R8;

	if ((hr = SlideAC->GetSlideProperties(TDH_SLIDE_PROP_TAG_PIXEL_WIDTH_IN_MICROMETER, &res)) == S_OK)
	{
		//printf("mpp为：%f\r\n", res.dblVal);
		return res.dblVal;
	}

	return 0;
}




EXTERN_C __declspec(dllexport) int WINAPI GetImage(ITDHSlideAC* SlideAC, long X1, long Y1, long X2, long Y2, long Magnify,  unsigned char * point)
{
	HRESULT hr;
	CComQIPtr<ITDHBitmapImage> TDHBitmap;
	TDHBitmap.CoCreateInstance(CLSID_TDHBitmapImage);
	unsigned char* bits;
	if ((hr = SlideAC->GetImage(X1, Y1, X2, Y2, Magnify, 0, 1, 2, TDHBitmap)) == S_OK)
	{
		long w, h, s;
		TDHBitmap->get_Height(&h);
		TDHBitmap->get_Width(&w);
		TDHBitmap->get_Stride(&s);
		TDHBitmap->LockBits(&bits);
		int l = h*s;
		memcpy(point, bits, l);

		TDHBitmap->UnlockBits(bits);

		TDHBitmap.Release();

		return l;
	}
	return 0;
}



EXTERN_C __declspec(dllexport) int WINAPI GetAssociatedImages(ITDHSlideAC* SlideAC, long ImageType,unsigned char * point)
{
	HRESULT hr;
	CComQIPtr<ITDHBitmapImage> TDHBitmap;
	TDHBitmap.CoCreateInstance(CLSID_TDHBitmapImage);
	unsigned char* bits;
	if ((hr = SlideAC->GetSingleImage(ImageType, &TDHBitmap)) == S_OK)
	{
		long ass_width, ass_height, s;
		TDHBitmap->get_Height(&ass_height);
		TDHBitmap->get_Width(&ass_width);
		TDHBitmap->get_Stride(&s);
		TDHBitmap->LockBits(&bits);
		int l = ass_height*s;
		memcpy(point, bits, l);
		TDHBitmap->UnlockBits(bits);
		TDHBitmap.Release();

		return s;
	}
	return 0;
}

//EXTERN_C __declspec(dllexport) int WINAPI GetAssociatedImagesProperties(ITDHSlideAC* SlideAC, long ImageType, long flag)
//{
//	HRESULT hr;
//	VARIANT res;
//	VariantInit(&res);
//	res.vt = 19;
//
//	if ((hr = SlideAC->GetSingleImageProperties(100,500, &res)) == S_OK)
//	{
//		printf("dll中：%d\r\n", res.intVal);
//		return res.intVal;
//	}
//
//	return 0;
//
//}




//
EXTERN_C __declspec(dllexport) BOOL WINAPI GetAssociatedImagesProperties(ITDHSlideAC* SlideAC, long ImageType, int* s, int* width, int* height)
{
	HRESULT hr;
	CComQIPtr<ITDHBitmapImage> TDHBitmap;
	TDHBitmap.CoCreateInstance(CLSID_TDHBitmapImage);
	long   ass_width, ass_height,stride;
	if ((hr = SlideAC->GetSingleImage(ImageType, &TDHBitmap)) == S_OK)
	{

		TDHBitmap->get_Width(&ass_width);
		TDHBitmap->get_Height(&ass_height);
		TDHBitmap->get_Stride(&stride);
		*s = stride;
		*height = ass_height;
		*width = ass_width;
		//printf("stride为：%d\r\n", stride);
		//printf("ass_height为：%d\r\n", ass_height);
		TDHBitmap.Release();
		return true;
	}
	else
	{
		return false;
	}
}

