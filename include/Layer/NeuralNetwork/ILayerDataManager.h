//=======================================
// レイヤー管理クラス
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_MANAGER_H__
#define __GRAVISBELL_I_NN_LAYER_MANAGER_H__

#include"../../Common/Guiddef.h"
#include"../../Common/ErrorCode.h"

#include"ILayerDLLManager.h"
#include"INNLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class ILayerDataManager
	{
	public:
		/** コンストラクタ */
		ILayerDataManager(){}
		/** デストラクタ */
		virtual ~ILayerDataManager(){}

	public:
		/** レイヤーデータの作成.	内部的に管理まで行う.
			@param	i_layerDLLManager	レイヤーDLL管理クラス.
			@param	i_typeCode			レイヤー種別コード
			@param	i_guid				新規作成するレイヤーデータのGUID
			@param	i_layerStructure	レイヤー構造
			@param	i_inputDataStruct	入力データ構造
			@param	o_pErrorCode		エラーコード格納先のアドレス. NULL指定可.
			@return
			typeCodeが存在しない場合、NULLを返す.
			既に存在するguidでtypeCodeも一致した場合、内部保有のレイヤーデータを返す.
			既に存在するguidでtypeCodeが異なる場合、NULLを返す. */
		virtual ILayerData* CreateLayerData(
			const ILayerDLLManager& i_layerDLLManager, const Gravisbell::GUID& i_typeCode,
			const Gravisbell::GUID& i_guid,
			const SettingData::Standard::IData& i_layerStructure,
			const IODataStruct& i_inputDataStruct,
			Gravisbell::ErrorCode* o_pErrorCode = NULL) = 0;
		
		/** レイヤーデータをバッファから作成.内部的に管理まで行う.
			@param	i_layerDLLManager	レイヤーDLL管理クラス.
			@param	i_typeCode			レイヤー種別コード
			@param	i_guid				新規作成するレイヤーデータのGUID
			@param	i_lpBuffer			読み取り用バッファ.
			@param	i_bufferSize		使用可能なバッファサイズ.
			@param	o_useBufferSize		実際に使用したバッファサイズ.
			@param	o_pErrorCode		エラーコード格納先のアドレス. NULL指定可.
			@return
			typeCodeが存在しない場合、NULLを返す.
			既に存在するguidでtypeCodeも一致した場合、内部保有のレイヤーデータを返す.
			既に存在するguidでtypeCodeが異なる場合、NULLを返す. */
		virtual ILayerData* CreateLayerData(
			const ILayerDLLManager& i_layerDLLManager,
			const Gravisbell::GUID& i_typeCode,
			const Gravisbell::GUID& i_guid,
			const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize,
			Gravisbell::ErrorCode* o_pErrorCode = NULL) = 0;


		/** レイヤーデータをGUID指定で取得する */
		virtual ILayerData* GetLayerData(const Gravisbell::GUID& i_guid) = 0;

		/** レイヤーデータ数を取得する */
		virtual U32 GetLayerDataCount() = 0;
		/** レイヤーデータを番号指定で取得する */
		virtual ILayerData* GetLayerDataByNum(U32 i_num) = 0;

		/** レイヤーデータをGUID指定で削除する */
		virtual Gravisbell::ErrorCode EraseLayerByGUID(const Gravisbell::GUID& i_guid) = 0;
		/** レイヤーデータを番号指定で削除する */
		virtual Gravisbell::ErrorCode EraseLayerByNum(U32 i_num) = 0;

		/** レイヤーデータをすべて削除する */
		virtual Gravisbell::ErrorCode ClearLayerData() = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif