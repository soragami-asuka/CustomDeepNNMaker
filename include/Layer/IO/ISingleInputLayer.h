//=======================================
// 単一入力レイヤー
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_INPUT_LAYER_H__
#define __GRAVISBELL_I_SINGLE_INPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"
#include"IInputLayer.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** レイヤーベース */
	class ISingleInputLayer : public virtual IInputLayer
	{
	public:
		/** コンストラクタ */
		ISingleInputLayer(){}
		/** デストラクタ */
		virtual ~ISingleInputLayer(){}

	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		virtual IODataStruct GetInputDataStruct()const = 0;

		/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
		virtual U32 GetInputBufferCount()const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif