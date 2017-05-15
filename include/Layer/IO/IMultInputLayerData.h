//=======================================
// 複数入力を持つレイヤーのレイヤーデータ
//=======================================
#ifndef __GRAVISBELL_I_MULT_INPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_MULT_INPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class IMultInputlayerData : public virtual ILayerData
	{
	public:
		/** コンストラクタ */
		IMultInputlayerData() : ILayerData(){}
		/** デストラクタ */
		virtual ~IMultInputlayerData(){}


		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データの数を取得する */
		virtual U32 GetInputDataCount()const = 0;

		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		virtual IODataStruct GetInputDataStruct(U32 i_dataNum)const = 0;

		/** 入力バッファ数を取得する. */
		virtual U32 GetInputBufferCount(U32 i_dataNum)const = 0;

	};

}	// IO
}	// Layer
}	// Gravisbell

#endif