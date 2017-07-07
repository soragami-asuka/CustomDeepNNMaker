//=======================================
// 入出力信号データレイヤー(逐次処理)
//=======================================
#ifndef __GRAVISBELL_I_IO_DATA_SEQUENTIAL_LAYER_H__
#define __GRAVISBELL_I_IO_DATA_SEQUENTIAL_LAYER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"

#include"IIODataLayer_base.h"


namespace Gravisbell {
namespace Layer {
namespace IOData {

	/** 入出力データレイヤー */
	class IIODataSequentialLayer : public IIODataLayer_base
	{
	public:
		/** コンストラクタ */
		IIODataSequentialLayer(){}
		/** デストラクタ */
		virtual ~IIODataSequentialLayer(){}

	public:
		/** データを追加する.
			@param	lpData	データ一組の配列. [GetBufferSize()の戻り値]の要素数が必要. */
		virtual ErrorCode SetData(U32 dataNo, const float lpData[]) = 0;
		/** データを追加する.
			@param	lpData	データ一組の配列. [GetBufferSize()の戻り値]の要素数が必要. 0～255の値. 内部的には0.0～1.0に変換される. */
		virtual ErrorCode SetData(U32 dataNo, const BYTE lpData[]) = 0;
	};

}	// IOData
}	// Layer
}	// Gravisbell

#endif