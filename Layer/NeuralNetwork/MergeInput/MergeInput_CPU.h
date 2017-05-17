//======================================
// 活性関数レイヤー
// CPU処理用
//======================================
#ifndef __MergeInput_CPU_H__
#define __MergeInput_CPU_H__

#include"stdafx.h"

#include"MergeInput_DATA.hpp"
#include"MergeInput_FUNC.hpp"
#include"MergeInput_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class MergeInput_CPU : public MergeInput_Base
{
private:
	// データ本体
	class MergeInput_LayerData_CPU& layerData;

	// 入出力バッファ
	std::vector<F32>				lpOutputBuffer;		/**< 出力バッファ [バッチ数 * 出力信号数] */
	std::vector<std::vector<F32>>	lpDInputBuffer;		/**< 入力誤差差分 [入力データ数][バッチ数 * 入力信号数] */

	std::vector<F32*>				lppBatchOutputBuffer;		/**< バッチ処理用出力バッファ <バッチ数> */
	std::vector<std::vector<F32*>>	lppBatchDInputBuffer;		/**< バッチ処理用入力誤差バッファ <入力データ数><バッチ数> */


	// Get関数を使うと処理負荷がかさむので一時保存用. PreCalculateで値を格納.
	std::vector<U32>	lpInputBufferCount;		/**< 入力バッファ数 */
	U32					outputBufferCount;		/**< 出力バッファ数 */

	// 演算時の入力データ
	std::vector<std::vector<CONST_BATCH_BUFFER_POINTER>>	m_lppInputBuffer;		/**< 演算時の入力データ[入力データ数][バッチサイズ] */
	std::vector<CONST_BATCH_BUFFER_POINTER>					m_lppDOutputBufferPrev;	/**< 入力誤差計算時の出力誤差データ */


public:
	/** コンストラクタ */
	MergeInput_CPU(Gravisbell::GUID guid, class MergeInput_LayerData_CPU& i_layerData);
	/** デストラクタ */
	virtual ~MergeInput_CPU();

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
	MergeInput_LayerData_Base& GetLayerData();
	const MergeInput_LayerData_Base& GetLayerData()const;


public:
	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode PreProcessLearn(unsigned int batchSize);

	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessCalculate(unsigned int batchSize);


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data);
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessCalculateLoop();


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[]);

	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;

public:
	//================================
	// 学習処理
	//================================
	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GetDInputBuffer(U32 i_dataNum)const;
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode GetDInputBuffer(U32 i_dataNum, BATCH_BUFFER_POINTER o_lpDInputBuffer)const;

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif