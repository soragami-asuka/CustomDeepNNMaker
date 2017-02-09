// 以下の ifdef ブロックは DLL からのエクスポートを容易にするマクロを作成するための 
// 一般的な方法です。この DLL 内のすべてのファイルは、コマンド ラインで定義された NNLAYERDLLMANAGER_EXPORTS
// シンボルを使用してコンパイルされます。このシンボルは、この DLL を使用するプロジェクトでは定義できません。
// ソースファイルがこのファイルを含んでいる他のプロジェクトは、 
// NNLAYERDLLMANAGER_API 関数を DLL からインポートされたと見なすのに対し、この DLL は、このマクロで定義された
// シンボルをエクスポートされたと見なします。
#ifdef NNLAYERDLLMANAGER_EXPORTS
#define NNLAYERDLLMANAGER_API __declspec(dllexport)
#else
#define NNLAYERDLLMANAGER_API __declspec(dllimport)
#endif

#include"INNLayerDLLManager.h"

namespace CustomDeepNNLibrary
{
	extern "C" NNLAYERDLLMANAGER_API INNLayerDLLManager* CreateLayerDLLManager();
}
