//=======================================
// 単独入力を持つレイヤーのレイヤーデータ
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_INPUT_LAYER_DATA_H__
#define __GRAVISBELL_I_SINGLE_INPUT_LAYER_DATA_H__

#include"../ILayerData.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	class ISingleInputLayerData : public virtual ILayerData
	{
	public:
		/** コンストラクタ */
		ISingleInputLayerData() : ILayerData(){}
		/** デストラクタ */
		virtual ~ISingleInputLayerData(){}


		//===========================
		// 入力レイヤー関連
		//===========================
	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		virtual IODataStruct GetInputDataStruct()const = 0;

		/** 入力バッファ数を取得する. */
		virtual U32 GetInputBufferCount()const = 0;

	};

}	// IOD
}	// Layer
}	// Gravisbell

#endif