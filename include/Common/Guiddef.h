//=======================================
// àÍî ê›íË
//=======================================
#ifndef __GRAVISBELL_GUIDDEF_H__
#define __GRAVISBELL_GUIDDEF_H__

#ifdef WIN32
#include<guiddef.h>
#else
#include"Common.h"
#endif

namespace Gravisbell {

#ifdef WIN32
	typedef ::GUID GUID;
#else
	typedef struct _GUID {
		U32 Data1;
		U16 Data2;
		U16 Data3;
		U08 Data4[ 8 ];
	} GUID;
#endif


}	// Gravisbell


#endif // __GRAVISBELL_COMMON_H__