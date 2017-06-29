//======================================
// 活性関レイヤー
//======================================
#ifndef __POOLING_BASE_H__
#define __POOLING_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"Pooling_DATA.hpp"

#include"Pooling_LayerData_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Pooling_Base : public INNSingle2SingleLayer
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤー識別用のGUID */
		IODataStruct inputDataStruct;	/**< 入力データ構造 */
		IODataStruct outputDataStruct;	/**< 出力データ構造 */

		SettingData::Standard::IData* pLearnData;	/**< 学習設定を定義したコンフィグクラス */

		unsigned int batchSize;	/**< バッチサイズ */

	public:
		/** コンストラクタ */
		Pooling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

		/** デストラクタ */
		virtual ~Pooling_Base();

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


		//===========================
		// レイヤーデータ関連
		//===========================
	public:
		/** レイヤーデータを取得する */
		virtual Pooling_LayerData_Base& GetLayerData() = 0;
		virtual const Pooling_LayerData_Base& GetLayerData()const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
