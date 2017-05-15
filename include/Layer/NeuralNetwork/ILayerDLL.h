//=======================================
// レイヤーDLLクラス
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_DLL_H__
#define __GRAVISBELL_I_NN_LAYER_DLL_H__

#include<guiddef.h>

#include"../../Common/VersionCode.h"

#include"INNLayer.h"
#include"../ILayerData.h"

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

		//==============================
		// レイヤー構造作成
		//==============================
		/** レイヤー構造設定を作成する */
		virtual SettingData::Standard::IData* CreateLayerStructureSetting(void)const = 0;
		/** レイヤー構造設定を作成する
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@param o_useBufferSize 実際に読み込んだバッファサイズ
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)const = 0;


		//==============================
		// 学習設定作成
		//==============================
		/** レイヤー学習設定を作成する */
		virtual SettingData::Standard::IData* CreateLearningSetting(void)const = 0;
		/** レイヤー学習設定を作成する
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@param o_useBufferSize 実際に読み込んだバッファサイズ
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)const = 0;


		//==============================
		// レイヤー作成
		//==============================
		/** レイヤーデータを作成.
			GUIDは自動割り当て.
			@param	i_layerStructure	レイヤー構造.
			@param	i_inputDataStruct	入力データ構造. */
		virtual ILayerData* CreateLayerData(const SettingData::Standard::IData& i_layerStructure, const IODataStruct& i_inputDataStruct)const = 0;
		/** レイヤーデータを作成
			@param guid	作成レイヤーのGUID
			@param	i_layerStructure	レイヤー構造.
			@param	i_inputDataStruct	入力データ構造. */
		virtual ILayerData* CreateLayerData(const Gravisbell::GUID& guid, const SettingData::Standard::IData& i_layerStructure, const IODataStruct& i_inputDataStruct)const = 0;
		
		/** レイヤーを作成.
			GUIDは自動割り当て.
			@param	i_lpBuffer		読み取り用バッファ.
			@param	i_bufferSize	使用可能なバッファサイズ.
			@param	o_useBufferSize	実際に使用したバッファサイズ. */
		virtual ILayerData* CreateLayerDataFromBuffer(const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)const = 0;
		/** レイヤーを作成
			@param guid	作成レイヤーのGUID */
		virtual ILayerData* CreateLayerDataFromBuffer(const Gravisbell::GUID& guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif