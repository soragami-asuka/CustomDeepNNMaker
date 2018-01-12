//======================================
// 活性関数レイヤー
// CPU処理用
//======================================
#ifndef __MergeMax_CPU_H__
#define __MergeMax_CPU_H__

#include"stdafx.h"

#include"MergeMax_DATA.hpp"
#include"MergeMax_FUNC.hpp"
#include"MergeMax_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class MergeMax_CPU : public MergeMax_Base
{
private:
	// データ本体
	class MergeMax_LayerData_CPU& layerData;

	// 入出力バッファ
	std::vector<std::vector<CONST_BATCH_BUFFER_POINTER>>	lppBatchInputBuffer;		/**< 演算時の入力データ[入力データ数][バッチサイズ] */
	std::vector<std::vector<F32*>>							lppBatchDInputBuffer;		/**< バッチ処理用入力誤差バッファ <入力データ数><バッチ数> */
	std::vector<F32*>										lppBatchOutputBuffer;		/**< バッチ処理用出力バッファ <バッチ数> */
	std::vector<CONST_BATCH_BUFFER_POINTER>					lppBatchDOutputBuffer;		/**< 入力誤差計算時の出力誤差データ */


	// Get関数を使うと処理負荷がかさむので一時保存用. PreCalculateで値を格納.
	std::vector<U32>	lpInputBufferCount;		/**< 入力バッファ数 */
	U32					outputBufferCount;		/**< 出力バッファ数 */


public:
	/** コンストラクタ */
	MergeMax_CPU(Gravisbell::GUID guid, class MergeMax_LayerData_CPU& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	/** デストラクタ */
	virtual ~MergeMax_CPU();

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
	MergeMax_LayerData_Base& GetLayerData();
	const MergeMax_LayerData_Base& GetLayerData()const;


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
	ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer);

public:
	//================================
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif