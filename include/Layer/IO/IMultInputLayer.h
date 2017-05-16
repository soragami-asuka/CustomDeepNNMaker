//=======================================
// 複数入力レイヤー
//=======================================
#ifndef __GRAVISBELL_I_MULT_INPUT_LAYER_H__
#define __GRAVISBELL_I_MULT_INPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"
#include"IInputLayer.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** レイヤーベース */
	class IMultInputLayer : public virtual IInputLayer
	{
	public:
		/** コンストラクタ */
		IMultInputLayer(){}
		/** デストラクタ */
		virtual ~IMultInputLayer(){}

	public:
		/** 入力データの数を取得する */
		virtual U32 GetInputDataCount()const = 0;

		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		virtual IODataStruct GetInputDataStruct(U32 i_dataNum)const = 0;

		/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
		virtual U32 GetInputBufferCount(U32 i_dataNum)const = 0;

		/** 学習差分を取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
			@return	誤差差分配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetDInputBuffer(U32 i_dataNum)const = 0;
		/** 学習差分を取得する.
			@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要.
			@return 成功した場合0 */
		virtual ErrorCode GetDInputBuffer(U32 i_dataNum, BATCH_BUFFER_POINTER o_lpDInputBuffer)const = 0;
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif