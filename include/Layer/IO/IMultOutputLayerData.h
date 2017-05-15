//=======================================
// 複数出力を持つレイヤーのレイヤーデータ
//=======================================
#ifndef __GRAVISBELL_I_MULT_OUTPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_MULT_OUTPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class IMultOutputLayerData : public ILayerData
	{
	public:
		/** コンストラクタ */
		IMultOutputLayerData() : ILayerData(){}
		/** デストラクタ */
		virtual ~IMultOutputLayerData(){}


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データの数を取得する */
		virtual U32 GetOutputDataCount()const = 0;

		/** 出力データ構造を取得する */
		virtual IODataStruct GetOutputDataStruct(U32 i_dataNum)const = 0;

		/** 出力バッファ数を取得する */
		virtual U32 GetOutputBufferCount(U32 i_dataNum)const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif