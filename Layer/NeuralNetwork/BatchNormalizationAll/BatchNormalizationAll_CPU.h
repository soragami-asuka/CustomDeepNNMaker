//======================================
// 活性関数レイヤー
// CPU処理用
//======================================
#ifndef __BatchNormalizationAll_CPU_H__
#define __BatchNormalizationAll_CPU_H__

#include"stdafx.h"

#include"BatchNormalizationAll_DATA.hpp"
#include"BatchNormalizationAll_FUNC.hpp"
#include"BatchNormalizationAll_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class BatchNormalizationAll_CPU : public BatchNormalizationAll_Base
{
private:
	// データ本体
	class BatchNormalizationAll_LayerData_CPU& layerData;
//	BatchNormalizationAll::LearnDataStructure learnData;

	// 入出力バッファ
	std::vector<CONST_BATCH_BUFFER_POINTER> lppBatchInputBuffer;	/**< 演算時の入力データ */
	std::vector<BATCH_BUFFER_POINTER>		lppBatchOutputBuffer;	/**< バッチ処理用出力バッファ <バッチ数> */
	std::vector<BATCH_BUFFER_POINTER>		lppBatchDInputBuffer;	/**< 入力誤差計算時の出力誤差データ */
	std::vector<CONST_BATCH_BUFFER_POINTER> lppBatchDOutputBuffer;	/**< 入力誤差計算時の出力誤差データ */

	// Get関数を使うと処理負荷がかさむので一時保存用. PreCalculateで値を格納.
	U32 inputBufferCount;				/**< 入力バッファ数 */
	U32 outputBufferCount;				/**< 出力バッファ数 */

	// 学習用のデータ
	bool onLearnMode;	/**< 学習処理中フラグ */
	U32 learnCount;		/**< 学習実行回数 */
	std::vector<F32> lpTmpMean;			/**< 平均値格納用の一時変数 */
	std::vector<F32> lpTmpVariance;		/**< 分散値格納用の一時変数 */

	// 演算処理用のバッファ
	std::vector<F32> lpDBias;	/**< バイアスの変化量 */
	std::vector<F32> lpDScale;	/**< スケールの変化量 */

public:
	/** コンストラクタ */
	BatchNormalizationAll_CPU(Gravisbell::GUID guid, class BatchNormalizationAll_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	/** デストラクタ */
	virtual ~BatchNormalizationAll_CPU();

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
	ILayerData& GetLayerData();
	const ILayerData& GetLayerData()const;


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
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
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