//=======================================
// レイヤーDLLクラス
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_DLL_H__
#define __GRAVISBELL_I_NN_LAYER_DLL_H__

#include<guiddef.h>

#include"../../Common/VersionCode.h"

#include"INNLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class ILayerDLL
	{
	public:
		/** コンストラクタ */
		ILayerDLL(){}
		/** デストラクタ */
		virtual ~ILayerDLL(){}

	public:
		/** レイヤー識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		virtual ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)const = 0;
		/** バージョンコードを取得する.
			@param o_versionCode	格納先バッファ
			@return 成功した場合0 */
		virtual ErrorCode GetVersionCode(VersionCode& o_versionCode)const = 0;


		/** レイヤー構造設定を作成する */
		virtual SettingData::Standard::IData* CreateLayerStructureSetting(void)const = 0;
		/** レイヤー構造設定を作成する
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@param o_useBufferSize 実際に読み込んだバッファサイズ
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)const = 0;


		/** レイヤー学習設定を作成する */
		virtual SettingData::Standard::IData* CreateLearningSetting(void)const = 0;
		/** レイヤー学習設定を作成する
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@param o_useBufferSize 実際に読み込んだバッファサイズ
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)const = 0;

		
		/** CPU処理用のレイヤーを作成.
			GUIDは自動割り当て. */
		virtual INNLayer* CreateLayerCPU()const = 0;
		/** CPU処理用のレイヤーを作成
			@param guid	作成レイヤーのGUID */
		virtual INNLayer* CreateLayerCPU(Gravisbell::GUID guid)const = 0;
		
		/** GPU処理用のレイヤーを作成.
			GUIDは自動割り当て. */
		virtual INNLayer* CreateLayerGPU()const = 0;
		/** GPU処理用のレイヤーを作成 */
		virtual INNLayer* CreateLayerGPU(Gravisbell::GUID guid)const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif