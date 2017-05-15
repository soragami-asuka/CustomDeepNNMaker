//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#ifndef __CONVOLUTION_BASE_H__
#define __CONVOLUTION_BASE_H__

#include<Layer/NeuralNetwork/INNSingleInputLayer.h>
#include<Layer/NeuralNetwork/INNSingleOutputLayer.h>

#include<vector>
#include<Layer/NeuralNetwork/INNSingleInputLayer.h>
#include<Layer/NeuralNetwork/INNSingleOutputLayer.h>

#include"Convolution_DATA.hpp"

#include"Convolution_LayerData_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< ニューロンに使用するデータ型. float or double */

	class Convolution_Base : public INNSingleInputLayer, public INNSingleOutputLayer
	{
	protected:
		Gravisbell::GUID guid;	/**< レイヤー識別用のGUID */

		SettingData::Standard::IData* pLearnData;		/**< 学習設定を定義したコンフィグクラス */
		Convolution::LearnDataStructure learnData;	/**< 学習設定 */

		U32 batchSize;	/**< バッチサイズ */

	public:
		/** コンストラクタ */
		Convolution_Base(Gravisbell::GUID guid);

		/** デストラクタ */
		virtual ~Convolution_Base();

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
		U32 GetBatchSize()const;


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
		U32 GetInputBufferCount()const;


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const;

		/** 出力バッファ数を取得する */
		U32 GetOutputBufferCount()const;


		//===========================
		// 固有関数
		//===========================
	public:


		//===========================
		// レイヤーデータ関連
		//===========================
	public:
		/** レイヤーデータを取得する */
		virtual Convolution_LayerData_Base& GetLayerData() = 0;
		virtual const Convolution_LayerData_Base& GetLayerData()const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
