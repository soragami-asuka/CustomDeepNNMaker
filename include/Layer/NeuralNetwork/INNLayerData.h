//=======================================
// ニューラルネットワークのレイヤーに関するデータを取り扱うインターフェース
// バッファなどを管理する.
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_DATA_H__
#define __GRAVISBELL_I_NN_LAYER_DATA_H__

#include"../ILayerData.h"

#include"./INNLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INNLayerData : public ILayerData
	{
	public:
		/** コンストラクタ */
		INNLayerData() : ILayerData(){}
		/** デストラクタ */
		virtual ~INNLayerData(){}

		
		//===========================
		// レイヤー設定
		//===========================
	public:
		/** レイヤーの設定情報を取得する */
		virtual const SettingData::Standard::IData* GetLayerStructure()const = 0;


		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		virtual IODataStruct GetInputDataStruct()const = 0;

		/** 入力バッファ数を取得する. */
		virtual U32 GetInputBufferCount()const = 0;


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データ構造を取得する */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** 出力バッファ数を取得する */
		virtual unsigned int GetOutputBufferCount()const = 0;


	public:
		//===========================
		// レイヤー作成
		//===========================
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		virtual INNLayer* CreateLayer(const Gravisbell::GUID& guid) = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif