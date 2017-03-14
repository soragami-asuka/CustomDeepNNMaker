#ifdef IODataLayer_EXPORTS
#define IODataLayer_API __declspec(dllexport)
#else
#define IODataLayer_API __declspec(dllimport)
#endif

#include"Layer/IOData/IIODataLayer.h"


/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
	@param bufferSize	バッファのサイズ.※float型配列の要素数.
	@return	入力信号データレイヤーのアドレス */
extern "C" IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPU(Gravisbell::IODataStruct ioDataStruct);
/** 入力信号データレイヤーを作成する.CPU制御
	@param guid			レイヤーのGUID.
	@param bufferSize	バッファのサイズ.※float型配列の要素数.
	@return	入力信号データレイヤーのアドレス */
extern "C" IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPUwithGUID(GUID guid, Gravisbell::IODataStruct ioDataStruct);

/** 入力信号データレイヤーを作成する.GPU制御
	@param guid			レイヤーのGUID.
	@param bufferSize	バッファのサイズ.※float型配列の要素数.
	@return	入力信号データレイヤーのアドレス */
extern "C" IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerGPU(GUID guid, Gravisbell::IODataStruct ioDataStruct);
