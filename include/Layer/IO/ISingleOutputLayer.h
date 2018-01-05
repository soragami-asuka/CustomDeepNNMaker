//=======================================
// 単一出力レイヤー
//=======================================
#ifndef __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_SINGLE_OUTPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"
#include"IOutputLayer.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** レイヤーベース */
	class ISingleOutputLayer : public virtual IOutputLayer
	{
	public:
		/** コンストラクタ */
		ISingleOutputLayer(){}
		/** デストラクタ */
		virtual ~ISingleOutputLayer(){}

	public:
		/** 出力データ構造を取得する.
			@return	出力データ構造 */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** 出力バッファ数を取得する. byte数では無くデータの数なので注意 */
		virtual U32 GetOutputBufferCount()const = 0;
	};

}	// IO
}	// Layer
}	// GravisBell

#endif