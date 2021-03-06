//======================================
// 全結合ニューラルネットワークのレイヤーデータ
//======================================
#ifndef __FULLYCONNECT_ACTIVATION_DATA_BASE_H__
#define __FULLYCONNECT_ACTIVATION_DATA_BASE_H__

#include<Layer/NeuralNetwork/INNLayerData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

#include<vector>

#include"FullyConnect_Activation_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class FullyConnect_Activation_LayerData_Base : public INNLayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤーデータ識別用のGUID */

		IODataStruct inputDataStruct;	/**< 入力データ構造 */

		SettingData::Standard::IData* pLayerStructure;	/**< レイヤー構造を定義したコンフィグクラス */
		FullyConnect_Activation::LayerStructure layerStructure;	/**< レイヤー構造 */


		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		FullyConnect_Activation_LayerData_Base(const Gravisbell::GUID& guid);
		/** デストラクタ */
		virtual ~FullyConnect_Activation_LayerData_Base();


		//===========================
		// 共通処理
		//===========================
	public:
		/** レイヤー固有のGUIDを取得する */
		Gravisbell::GUID GetGUID(void)const;

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::GUID GetLayerCode(void)const;


		//===========================
		// レイヤー設定
		//===========================
	public:
		/** 設定情報を設定 */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		U32 GetUseBufferByteCount()const;


		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct()const;

		/** 入力バッファ数を取得する. */
		U32 GetInputBufferCount()const;


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const;

		/** 出力バッファ数を取得する */
		unsigned int GetOutputBufferCount()const;
		

		//===========================
		// 固有関数
		//===========================
	public:
		/** ニューロン数を取得する */
		U32 GetNeuronCount()const;
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif