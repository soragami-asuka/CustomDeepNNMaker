//=======================================
// 入出力信号データレイヤー(逐次処理)
//=======================================
#ifndef __GRAVISBELL_I_IO_DATA_SEQUENTIAL_LAYER_H__
#define __GRAVISBELL_I_IO_DATA_SEQUENTIAL_LAYER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"

#include"../IO/ISingleOutputLayer.h"
#include"../IO/ISingleInputLayer.h"
#include"IDataLayer.h"


namespace Gravisbell {
namespace Layer {
namespace IOData {

	/** 入出力データレイヤー */
	class IIODataSequentialLayer : public IO::ISingleOutputLayer, public IO::ISingleInputLayer, public IDataLayer
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

		/** 学習誤差を計算する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要 */
		virtual ErrorCode CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;

		/** 誤差の値を取得する.
			CalculateLearnError()を1回以上実行していない場合、正常に動作しない.
			@param	o_min	最小誤差.
			@param	o_max	最大誤差.
			@param	o_ave	平均誤差.
			@param	o_ave2	平均二乗誤差.
			@param	o_crossEntropy	クロスエントロピー*/
		virtual ErrorCode GetCalculateErrorValue(F32& o_max, F32& o_ave, F32& o_ave2, F32& o_crossEntropy) = 0;

		/** 詳細な誤差の値を取得する.
			各入出力の値毎に誤差を取る.
			CalculateLearnError()を1回以上実行していない場合、正常に動作しない.
			各配列の要素数は[GetBufferCount()]以上である必要がある.
			@param	o_lpMin		最小誤差.
			@param	o_lpMax		最大誤差.
			@param	o_lpAve		平均誤差.
			@param	o_lpAve2	平均二乗誤差. */
		virtual ErrorCode GetCalculateErrorValueDetail(F32 o_lpMax[], F32 o_lpAve[], F32 o_lpAve2[]) = 0;


		/** 学習差分を取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
			@return	誤差差分配列の先頭ポインタ */
		virtual CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const = 0;
		/** 学習差分を取得する.
			@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要.
			@return 成功した場合0 */
		virtual ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const = 0;
	};

}	// IOData
}	// Layer
}	// Gravisbell

#endif