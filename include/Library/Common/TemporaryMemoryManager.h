//===============================================
// ç≈ìKâªÉãÅ[É`Éì
//===============================================
#ifndef __GRAVISBELL_LIBRARY_NN_TEMPORARY_MEMORY_MANAGER_H__
#define __GRAVISBELL_LIBRARY_NN_TEMPORARY_MEMORY_MANAGER_H__

#ifdef TemporaryMemoryManager_EXPORTS
#define TemporaryMemoryManager_API __declspec(dllexport)
#else
#define TemporaryMemoryManager_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#endif
#endif

#include"Common/ITemporaryMemoryManager.h"

namespace Gravisbell {
namespace Common {

	extern TemporaryMemoryManager_API ITemporaryMemoryManager* CreateTemporaryMemoryManagerCPU();
	extern TemporaryMemoryManager_API ITemporaryMemoryManager* CreateTemporaryMemoryManagerGPU();

}	// Layer
}	// Gravisbell


#endif
