//=======================================
// レイヤーDLL関数
//=======================================
#ifndef __GRAVISBELL_NN_LAYER_FUNCTION_H__
#define __GRAVISBELL_NN_LAYER_FUNCTION_H__

#include"../../Common/Guiddef.h"
#include"../../Common/VersionCode.h"
#include"../../Common/ErrorCode.h"

#include"ILayerDLLManager.h"
#include"INNLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** レイヤー識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	typedef ErrorCode (*FuncGetLayerCode)(Gravisbell::GUID& o_layerCode);
	/** バージョンコードを取得する.
		@param o_versionCode	格納先バッファ
		@return 成功した場合0 */
	typedef ErrorCode (*FuncGetVersionCode)(VersionCode& o_versionCode);


	/** レイヤー構造設定を作成する */
	typedef SettingData::Standard::IData* (*FuncCreateLayerStructureSetting)(void);
	/** レイヤー構造設定を作成する
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@param o_useBufferSize 実際に読み込んだバッファサイズ
		@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
	typedef SettingData::Standard::IData* (*FuncCreateLayerStructureSettingFromBuffer)(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


	/** 学習設定を作成する */
	typedef SettingData::Standard::IData* (*FuncCreateLayerLearningSetting)(void);
	/** 学習設定を作成する
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@param o_useBufferSize 実際に読み込んだバッファサイズ
		@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
	typedef SettingData::Standard::IData* (*FuncCreateLayerLearningSettingFromBuffer)(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


	/** CPU処理用のレイヤーを作成 */
	typedef INNLayer* (*FuncCreateLayerCPU)(Gravisbell::GUID guid, const ILayerDLLManager* pLayerDLLManager);

	/** GPU処理用のレイヤーを作成 */
	typedef INNLayer* (*FuncCreateLayerGPU)(Gravisbell::GUID guid, const ILayerDLLManager* pLayerDLLManager);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif