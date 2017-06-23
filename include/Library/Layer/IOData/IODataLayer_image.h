#ifdef IODataLayer_image_EXPORTS
#define IODataLayer_image_API __declspec(dllexport)
#else
#define IODataLayer_image_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#pragma comment(lib, "Gravisbell.Layer.IOData.IODataLayer_image.lib")
#endif
#endif

#include"../../../Layer/IOData/IIODataLayer_image.h"
#include"../../../Common/Guiddef.h"
#include"../../../Common/VersionCode.h"


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
extern IODataLayer_image_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode);

/** バージョンコードを取得する.
  * @param  o_versionCode	格納先バッファ.
  * @return 成功した場合0. 
  */
extern IODataLayer_image_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode);


/** レイヤー学習設定を作成する */
extern IODataLayer_image_API SettingData::Standard::IData* CreateLearningSetting(void);
/** レイヤー学習設定を作成する
	@param i_lpBuffer	読み込みバッファの先頭アドレス.
	@param i_bufferSize	読み込み可能バッファのサイズ.
	@param o_useBufferSize 実際に読み込んだバッファサイズ
	@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
extern IODataLayer_image_API SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


//======================================
// CPU処理
//======================================

/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerCPU(Gravisbell::U32 i_dataCount, U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch);
/** 入力信号データレイヤーを作成する.CPU制御
	@param guid			レイヤーのGUID.
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerCPUwithGUID(Gravisbell::GUID guid, Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch);


//======================================
// GPU処理
//======================================
/** 入力信号データレイヤーを作成する.GPU制御
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerGPU(Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch);
/** 入力信号データレイヤーを作成する.GPU制御
	@param guid			レイヤーのGUID.
	@param ioDataStruct	入出力データ構造.
	@return	入力信号データレイヤーのアドレス */
extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerGPUwithGUID(Gravisbell::GUID guid, Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch);




}	// IOData
}	// Layer
}	// Gravisbell