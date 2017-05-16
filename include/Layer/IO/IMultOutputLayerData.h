//=======================================
// 複数出力を持つレイヤーのレイヤーデータ
//=======================================
#ifndef __GRAVISBELL_I_MULT_OUTPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_MULT_OUTPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class IMultOutputLayerData : public virtual ILayerData
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
		/** 出力データの出力先レイヤー数. */
		virtual U32 GetOutputToLayerCount()const = 0;

		/** 出力データ構造を取得する */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** 出力バッファ数を取得する */
		virtual U32 GetOutputBufferCount()const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif