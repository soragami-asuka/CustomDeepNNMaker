#ifdef IODataSequentialLayer_EXPORTS
#define IODataSequentialLayer_API __declspec(dllexport)
#else
#define IODataSequentialLayer_API __declspec(dllimport)
#endif

#include"Layer/IOData/IIODataSequentialLayer.h"
#include"Common/Guiddef.h"
#include"Common/VersionCode.h"

namespace Gravisbell {
namespace Layer {
namespace IOData {

//======================================
// 共通部分
//======================================

/** レイヤーの識別コードを取得する.
  * @param  o_layerCode		格納先バッファ.
  * @return 成功した場合0. 
  */
extern IODataSequentialLayer_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode);

/** バージョンコードを取得する.
  * @param  o_versionCode	格納先バッファ.
  * @return 成功した場合0. 
  */
extern IODataSequentialLayer_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode);


/** レイヤー学習設定を作成する */
extern IODataSequentialLayer_API SettingData::Standard::IData* CreateLearningSetting(void);
/** レイヤー学習設定を作成する
	@param i_lpBuffer	読み込みバッファの先頭アドレス.
	@param i_bufferSize	読み込み可能バッファのサイズ.
	@param o_useBufferSize 実際に読み込んだバッファサイズ
	@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
extern IODataSequentialLayer_API SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


//======================================
// CPU処理
//======================================

/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerCPU(Gravisbell::IODataStruct ioDataStruct);
/** 入力信号データレイヤーを作成する.CPU制御
	@param guid			レイヤーのGUID.
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerCPUwithGUID(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);


//======================================
// GPU処理
// データをホストに確保
//======================================
/** 入力信号データレイヤーを作成する.GPU制御
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPU_host(Gravisbell::IODataStruct ioDataStruct);
/** 入力信号データレイヤーを作成する.GPU制御
	@param guid			レイヤーのGUID.
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPUwithGUID_host(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);


//======================================
// GPU処理
// データをデバイスに確保
//======================================
/** 入力信号データレイヤーを作成する.GPU制御
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPU_device(Gravisbell::IODataStruct ioDataStruct);
/** 入力信号データレイヤーを作成する.GPU制御
	@param guid			レイヤーのGUID.
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPUwithGUID_device(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);



}	// IOData
}	// Layer
}	// Gravisbell