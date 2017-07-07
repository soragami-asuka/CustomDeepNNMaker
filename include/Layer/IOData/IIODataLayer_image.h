//=======================================
// 入出力信号データレイヤー
//=======================================
#ifndef __GRAVISBELL_I_IO_DATA_LAYER_H__
#define __GRAVISBELL_I_IO_DATA_LAYER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"

#include"IIODataLayer_base.h"


namespace Gravisbell {
namespace Layer {
namespace IOData {

	/** 入出力データレイヤー */
	class IIODataLayer_image : public IIODataLayer_base
	{
	public:
		/** コンストラクタ */
		IIODataLayer_image(){}
		/** デストラクタ */
		virtual ~IIODataLayer_image(){}

	public:
		/** データを追加する.
			@param	lpData	データ一組の配列. [GetBufferSize()の戻り値]の要素数が必要. */
		virtual ErrorCode SetData(U32 i_dataNum, const BYTE lpData[], U32 i_lineLength) = 0;
		/** データを番号指定で取得する.
			@param num		取得する番号
			@param o_LpData データの格納先配列. [GetBufferSize()の戻り値]の要素数が必要. */
		virtual ErrorCode GetDataByNum(U32 num, BYTE o_lpData[])const = 0;


		/** バッチ処理データ番号リストを設定する.
			設定された値を元にGetDInputBuffer(),GetOutputBuffer()の戻り値が決定する.
			@param i_lpBatchDataNoList	設定するデータ番号リスト. [GetBatchSize()の戻り値]の要素数が必要 */
		virtual ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[]) = 0;

	};

}	// IOData
}	// Layer
}	// Gravisbell

#endif