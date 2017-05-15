//=======================================
// 単独出力を持つレイヤーのレイヤーデータ
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class ISingleOutputLayerData : public virtual ILayerData
	{
	public:
		/** コンストラクタ */
		ISingleOutputLayerData() : ILayerData(){}
		/** デストラクタ */
		virtual ~ISingleOutputLayerData(){}


		//===========================
		// 出力レイヤー関連
		//===========================
	public:
		/** 出力データ構造を取得する */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** 出力バッファ数を取得する */
		virtual U32 GetOutputBufferCount()const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif