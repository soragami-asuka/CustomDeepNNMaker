//=======================================
// レイヤーDLL関数
//=======================================
#ifndef __GRAVISBELL_NN_LAYER_FUNCTION_H__
#define __GRAVISBELL_NN_LAYER_FUNCTION_H__

#include"Common/Guiddef.h"
#include"Common/VersionCode.h"
#include"Common/ErrorCode.h"

#include"INNLayer.h"


namespace Gravisbell {
namespace NeuralNetwork {

	/** レイヤー識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	typedef Gravisbell::ErrorCode (*FuncGetLayerCode)(GUID& o_layerCode);
	/** バージョンコードを取得する.
		@param o_versionCode	格納先バッファ
		@return 成功した場合0 */
	typedef Gravisbell::ErrorCode (*FuncGetVersionCode)(Gravisbell::VersionCode& o_versionCode);


	/** レイヤー設定を作成する */
	typedef ILayerConfig* (*FuncCreateLayerConfig)(void);
	/** レイヤー設定を作成する
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@param o_useBufferSize 実際に読み込んだバッファサイズ
		@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
	typedef ILayerConfig* (*FuncCreateLayerConfigFromBuffer)(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


	/** CPU処理用のレイヤーを作成 */
	typedef INNLayer* (*FuncCreateLayerCPU)(GUID guid);

	/** GPU処理用のレイヤーを作成 */
	typedef INNLayer* (*FuncCreateLayerGPU)(GUID guid);

}	// NeuralNetwork
}	// Gravisbell

#endif