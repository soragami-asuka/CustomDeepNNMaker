//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#ifndef __FULLYCONNECT_ACTIVATION_BASE_H__
#define __FULLYCONNECT_ACTIVATION_BASE_H__

#include<Layer/NeuralNetwork/INNLayer.h>

#include<vector>

#include"FullyConnect_Activation_DATA.hpp"

#include"FullyConnect_Activation_LayerData_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class FullyConnect_Activation_Base : public Gravisbell::Layer::NeuralNetwork::INNLayer
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤー識別用のGUID */

		SettingData::Standard::IData* pLearnData;		/**< 学習設定を定義したコンフィグクラス */
		FullyConnect_Activation::LearnDataStructure learnData;	/**< 学習設定 */

		unsigned int batchSize;	/**< バッチサイズ */

	public:
		/** コンストラクタ */
		FullyConnect_Activation_Base(Gravisbell::GUID guid);

		/** デストラクタ */
		virtual ~FullyConnect_Activation_Base();

		//===========================
		// レイヤー共通
		//===========================
	public:
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKindBase(void)const;

		/** レイヤー固有のGUIDを取得する */
		Gravisbell::GUID GetGUID(void)const;

		/** レイヤーの種類識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::GUID GetLayerCode(void)const;

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		unsigned int GetBatchSize()const;


		//===========================
		// レイヤー設定
		//===========================
	public:
		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		unsigned int GetUseBufferByteCount()const;


		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		virtual IODataStruct GetInputDataStruct()const;

		/** 入力バッファ数を取得する. */
		unsigned int GetInputBufferCount()const;


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


		//===========================
		// レイヤーデータ関連
		//===========================
	public:
		/** レイヤーデータを取得する */
		virtual FullyConnect_Activation_LayerData_Base& GetLayerData() = 0;
		virtual const FullyConnect_Activation_LayerData_Base& GetLayerData()const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
