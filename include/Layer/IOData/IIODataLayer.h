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
	class IIODataLayer : public virtual IIODataLayer_base
	{
	public:
		/** コンストラクタ */
		IIODataLayer(){}
		/** デストラクタ */
		virtual ~IIODataLayer(){}

	public:
		/** データを追加する.
			@param	lpData	データ一組の配列. [GetBufferSize()の戻り値]の要素数が必要. */
		virtual ErrorCode AddData(const float lpData[]) = 0;
		/** データを番号指定で取得する.
			@param num		取得する番号
			@param o_LpData データの格納先配列. [GetBufferSize()の戻り値]の要素数が必要. */
		virtual ErrorCode GetDataByNum(unsigned int num, float o_lpData[])const = 0;
		/** データを番号指定で消去する */
		virtual ErrorCode EraseDataByNum(unsigned int num) = 0;

		/** データを全消去する.
			@return	成功した場合0 */
		virtual ErrorCode ClearData() = 0;

		/** バッチ処理データ番号リストを設定する.
			設定された値を元にGetDInputBuffer(),GetOutputBuffer()の戻り値が決定する.
			@param i_lpBatchDataNoList	設定するデータ番号リスト. [GetBatchSize()の戻り値]の要素数が必要 */
		virtual ErrorCode SetBatchDataNoList(const unsigned int i_lpBatchDataNoList[]) = 0;
	};

}	// IOData
}	// Layer
}	// Gravisbell

#endif