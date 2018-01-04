//======================================
// 活性関数レイヤー
// CPU処理用
//======================================
#ifndef __ACTIVATION_CPU_H__
#define __ACTIVATION_CPU_H__

#include"stdafx.h"

#include"Activation_DATA.hpp"
#include"Activation_FUNC.hpp"
#include"Activation_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class Activation_CPU : public Activation_Base
{
private:
	// データ本体
	class Activation_LayerData_CPU& layerData;

	// 入出力バッファ
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppInputBuffer;		/**< 演算時の入力データ */
	std::vector<BATCH_BUFFER_POINTER>		m_lppOutputBuffer;		/**< バッチ処理用出力バッファ <バッチ数> */
	std::vector<BATCH_BUFFER_POINTER>		m_lppDInputBuffer;		/**< 入力誤差データ */
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppDOutputBuffer;		/**< 入力誤差計算時の出力誤差データ */

	// Get関数を使うと処理負荷がかさむので一時保存用. PreCalculateで値を格納.
	U32 inputBufferCount;				/**< 入力バッファ数 */
	U32 outputBufferCount;				/**< 出力バッファ数 */

	std::vector<F32>						lpCalculateSum;	/**< 一時計算用のバッファ[z][y][x]のサイズを持つ */


	// 活性化関数
	F32 (Activation_CPU::*func_activation)(F32);
	F32 (Activation_CPU::*func_dactivation)(F32);

public:
	/** コンストラクタ */
	Activation_CPU(Gravisbell::GUID guid, class Activation_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	/** デストラクタ */
	virtual ~Activation_CPU();

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
	// 演算処理
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


protected:
	//================================
	// 活性化関数
	//================================
	// lenear系
	F32 func_activation_lenear(F32 x);
	F32 func_dactivation_lenear(F32 x);

	// sigmoid系
	F32 func_activation_sigmoid(F32 x);
	F32 func_dactivation_sigmoid(F32 x);

	F32 func_activation_sigmoid_crossEntropy(F32 x);
	F32 func_dactivation_sigmoid_crossEntropy(F32 x);

	// ReLU系
	F32 func_activation_ReLU(F32 x);
	F32 func_dactivation_ReLU(F32 x);

	// Leaky-ReLU系
	F32 func_activation_LeakyReLU(F32 x);
	F32 func_dactivation_LeakyReLU(F32 x);

	// tanh系
	F32 func_activation_tanh(F32 x);
	F32 func_dactivation_tanh(F32 x);

	// SoftMax系
	F32 func_activation_SoftMax(F32 x);

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif