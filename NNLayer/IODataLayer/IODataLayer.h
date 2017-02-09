#ifdef IODataLayer_EXPORTS
#define IODataLayer_API __declspec(dllexport)
#else
#define IODataLayer_API __declspec(dllimport)
#endif

#include"IIODataLayer.h"


/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
	@param bufferSize	バッファのサイズ.※float型配列の要素数.
	@return	入力信号データレイヤーのアドレス */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerCPU(CustomDeepNNLibrary::IODataStruct ioDataStruct);
/** 入力信号データレイヤーを作成する.CPU制御
	@param guid			レイヤーのGUID.
	@param bufferSize	バッファのサイズ.※float型配列の要素数.
	@return	入力信号データレイヤーのアドレス */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerCPUwithGUID(GUID guid, CustomDeepNNLibrary::IODataStruct ioDataStruct);

/** 入力信号データレイヤーを作成する.GPU制御
	@param guid			レイヤーのGUID.
	@param bufferSize	バッファのサイズ.※float型配列の要素数.
	@return	入力信号データレイヤーのアドレス */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerGPU(GUID guid, CustomDeepNNLibrary::IODataStruct ioDataStruct);
