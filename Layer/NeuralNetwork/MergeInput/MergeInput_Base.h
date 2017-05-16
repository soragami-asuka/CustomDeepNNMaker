//======================================
// 活性関レイヤー
//======================================
#ifndef __MergeInput_BASE_H__
#define __MergeInput_BASE_H__

#include<Layer/NeuralNetwork/INNMultInputLayer.h>
#include<Layer/NeuralNetwork/INNSingleOutputLayer.h>

#include<vector>

#include"MergeInput_DATA.hpp"

#include"MergeInput_LayerData_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class MergeInput_Base : public INNMultInputLayer, public INNSingleOutputLayer
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤー識別用のGUID */

		SettingData::Standard::IData* pLearnData;	/**< 学習設定を定義したコンフィグクラス */

		unsigned int batchSize;	/**< バッチサイズ */

	public:
		/** コンストラクタ */
		MergeInput_Base(Gravisbell::GUID guid);

		/** デストラクタ */
		virtual ~MergeInput_Base();

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
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データの数を取得する */
		U32 GetInputDataCount()const;

		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct(U32 i_dataNum)const;

		/** 入力バッファ数を取得する. */
		unsigned int GetInputBufferCount(U32 i_dataNum)const;


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


		//===========================
		// レイヤーデータ関連
		//===========================
	public:
		/** レイヤーデータを取得する */
		virtual MergeInput_LayerData_Base& GetLayerData() = 0;
		virtual const MergeInput_LayerData_Base& GetLayerData()const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
