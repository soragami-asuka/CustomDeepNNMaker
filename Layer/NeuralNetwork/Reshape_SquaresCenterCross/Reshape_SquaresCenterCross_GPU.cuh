//======================================
// 出力信号分割レイヤー
// GPU処理用
//======================================
#ifndef __RESHAPE_SQUARECENTERCROSS_GPU_H__
#define __RESHAPE_SQUARECENTERCROSS_GPU_H__

#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4819)
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#include"Reshape_SquaresCenterCross_DATA.hpp"
#include"Reshape_SquaresCenterCross_FUNC.hpp"
#include"Reshape_SquaresCenterCross_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class Reshape_SquaresCenterCross_GPU : public Reshape_SquaresCenterCross_Base
{
private:
	// データ本体
	class Reshape_SquaresCenterCross_LayerData_GPU& layerData;


	// Get関数を使うと処理負荷がかさむので一時保存用. PreCalculateで値を格納.
	U32 inputBufferCount;				/**< 入力バッファ数 */
	U32 outputBufferCount;				/**< 出力バッファ数 */

	std::vector<U32>						m_lpConvertTable;		/**< 変換テーブル(sqrt(x-1)*2+1)^2 */
	std::vector<U32*>						m_lppConvertTable;		/**< 変換テーブル<sqrt(x-1)*2+1><sqrt(x-1)*2+1> */

	// 入出力バッファ
	std::vector<F32>						m_lpInputBuffer_h;		/**< 入力バッファ(ホスト側) */
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppInputBuffer;		/**< 入力バッファ <バッチ数><入力信号数>  */

	std::vector<F32>						m_lpOutputBuffer_h;		/**< 出力バッファ <バッチ数 * 出力信号数> */
	std::vector<BATCH_BUFFER_POINTER>		m_lppOutputBuffer;		/**< 出力バッファ <バッチ数><出力信号数> */

	std::vector<F32>						m_lpDOutputBuffer_h;	/**< 出力誤差バッファ(ホスト側) */
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppDOutputBuffer;		/**< 出力誤差バッファ <バッチ数><出力信号数> */

	std::vector<F32>						m_lpDInputBuffer_h;		/**< 入力誤差バッファ(ホスト側) */
	std::vector<BATCH_BUFFER_POINTER>		m_lppDInputBuffer;		/**< 入力誤差バッファ <バッチ数><入力信号数> */


public:
	/** コンストラクタ */
	Reshape_SquaresCenterCross_GPU(Gravisbell::GUID guid, class Reshape_SquaresCenterCross_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	/** デストラクタ */
	virtual ~Reshape_SquaresCenterCross_GPU();

public:
	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 GetLayerKind()const;

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Initialize(void);


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Reshape_SquaresCenterCross_LayerData_Base& GetLayerData();
	const Reshape_SquaresCenterCross_LayerData_Base& GetLayerData()const;


public:
	//================================
	// 事前処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode PreProcessLearn();

	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessCalculate();


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessLoop();


public:
	//================================
	// 演算処理
	//================================
	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);

public:
	//================================
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列の[GetOutputDataCount()]配列
		直前の計算結果を使用する */
	ErrorCode CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif