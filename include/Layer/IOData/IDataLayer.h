//=======================================
// データレイヤー
//=======================================
#ifndef __GRAVISBELL_I_DATA_LAYER_H__
#define __GRAVISBELL_I_DATA_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

namespace Gravisbell {
namespace Layer {
namespace IOData {


	/** 入力データレイヤー */
	class IDataLayer
	{
	public:
		/** コンストラクタ */
		IDataLayer(){}
		/** デストラクタ */
		virtual ~IDataLayer(){}

	public:
		/** データの構造情報を取得する */
		virtual IODataStruct GetDataStruct()const = 0;

		/** データのバッファサイズを取得する.
			@return データのバッファサイズ.使用するfloat型配列の要素数. */
		virtual U32 GetBufferCount()const = 0;

		/** データ数を取得する */
		virtual U32 GetDataCount()const = 0;

	};

}	// IOData
}	// Layer
}	// Gravisbell

#endif