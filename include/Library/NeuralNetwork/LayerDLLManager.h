// 以下の ifdef ブロックは DLL からのエクスポートを容易にするマクロを作成するための 
// 一般的な方法です。この DLL 内のすべてのファイルは、コマンド ラインで定義された LayerDLLManager_EXPORTS
// シンボルを使用してコンパイルされます。このシンボルは、この DLL を使用するプロジェクトでは定義できません。
// ソースファイルがこのファイルを含んでいる他のプロジェクトは、 
// LayerDLLManager_API 関数を DLL からインポートされたと見なすのに対し、この DLL は、このマクロで定義された
// シンボルをエクスポートされたと見なします。
#ifdef LayerDLLManager_EXPORTS
#define LayerDLLManager_API __declspec(dllexport)
#else
#define LayerDLLManager_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#pragma comment(lib, "Gravisbell.NeuralNetwork.LayerDLLManager.lib")
#endif
#endif

#include"../../Common/ErrorCode.h"
#include"../../Common/VersionCode.h"

#include"../../Layer/NeuralNetwork/ILayerDLLManager.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	extern LayerDLLManager_API ILayerDLLManager* CreateLayerDLLManagerCPU();
	extern LayerDLLManager_API ILayerDLLManager* CreateLayerDLLManagerGPU();

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
