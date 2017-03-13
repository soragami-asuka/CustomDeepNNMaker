//=======================================
// 単一出力レイヤー
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_H__

#include"Common/ErrorCode.h"
#include"Common/IODataStruct.h"

#include"ILayerBase.h"
#include"IOutputLayer.h"

namespace Gravisbell {
namespace NeuralNetwork {

	/** レイヤーベース */
	class ISingleOutputLayer : public virtual IOutputLayer
	{
	public:
		/** コンストラクタ */
		ISingleOutputLayer(){}
		/** デストラクタ */
		virtual ~ISingleOutputLayer(){}

	public:
		/** 出力データ構造を取得する */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** 出力バッファ数を取得する. byte数では無くデータの数なので注意 */
		virtual U32 GetOutputBufferCount()const = 0;

		/** 出力データバッファを取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		virtual ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const = 0;
	};

}	// NeuralNetwork
}	// GravisBell

#endif