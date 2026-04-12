#include <stdio.h>
#include <windows.h>
#include "SlideAcWrapper.h"
#include "SlideAC.h"


void main(int argc, char **argv)
{
	
	ITDHSlideAC * ac = OpenSlide("C:\\Users\\dipath\\Desktop\\3DHISTECH SlideAC SDK 1.15.2\\Samples\\C\\WrapperDLLTest\\x64\\Release\\1");
	int w = GetLevelCount(ac);
	printf("%d\r\n", w);
	//CloseSlide();
	return;
}