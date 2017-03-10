//=======================================
// 単一入力レイヤー
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_INPUT_LAYER_H__
#define __GRAVISBELL_I_SINGLE_INPUT_LAYER_H__

#include"Common/ErrorCode.h"
#include"Common/IODataStruct.h"

#include"ILayerBase.h"
#include"IInputLayer.h"

namespace Gravisbell {
namespace NeuralNetwork {

	/** レイヤーベース */
	class ISingleInputLayer : public virtual IInputLayer
	{
	public:
		/** コンストラクタ */
		ISingleInputLayer(){}
		/** デストラクタ */
		virtual ~ISingleInputLayer(){}

	public:
		/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
		virtual unsigned int GetInputBufferCount()const = 0;

		/** 学習差分を取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
			@return	誤差差分配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const = 0;
		/** 学習差分を取得する.
			@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
		virtual ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const = 0;

	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		virtual IODataStruct GetInputDataStruct()const = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif