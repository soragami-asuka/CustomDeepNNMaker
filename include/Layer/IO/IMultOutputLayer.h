//=======================================
// 複数出力レイヤー
//=======================================
#ifndef __GRAVISBELL_I_MULT_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_MULT_OUTPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"
#include"IOutputLayer.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** レイヤーベース */
	class IMultOutputLayer : public virtual IOutputLayer
	{
	public:
		/** コンストラクタ */
		IMultOutputLayer(){}
		/** デストラクタ */
		virtual ~IMultOutputLayer(){}

	public:
		/** 出力データの数を取得する */
		virtual U32 GetOutputDataCount()const = 0;

		/** 出力データ構造を取得する.
			@return	出力データ構造 */
		virtual IODataStruct GetOutputDataStruct(U32 i_dataNum)const = 0;

		/** 出力バッファ数を取得する. byte数では無くデータの数なので注意 */
		virtual U32 GetOutputBufferCount(U32 i_dataNum)const = 0;

		/** 出力データバッファを取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
			@return 出力データ配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer(U32 i_dataNum)const = 0;
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		virtual ErrorCode GetOutputBuffer(U32 i_layerNum, BATCH_BUFFER_POINTER o_lpOutputBuffer)const = 0;
	};

}	// IO
}	// Layer
}	// GravisBell

#endif