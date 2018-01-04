//=======================================
// 単一入力を持つNNのレイヤー
//=======================================
#ifndef __GRAVISBELL_I_NN_SINGLE_TO_SINGLE_LAYER_H__
#define __GRAVISBELL_I_NN_SINGLE_TO_SINGLE_LAYER_H__

#include"../IO/ISingleInputLayer.h"
#include"../IO/ISingleOutputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** レイヤーベース */
	class INNSingle2SingleLayer : public virtual IO::ISingleInputLayer, public virtual IO::ISingleOutputLayer
	{
	public:
		/** コンストラクタ */
		INNSingle2SingleLayer() : ISingleInputLayer(), ISingleOutputLayer(){}
		/** デストラクタ */
		virtual ~INNSingle2SingleLayer(){}

	public:
		//==========================================
		// 演算処理.
		// 入出力はCPU側のメモリー
		//==========================================
		/** 演算処理を実行する.
			@param i_lppInputBuffer		入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0が返る */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;

		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		virtual ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;

		/** 学習処理を実行する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		virtual ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;


	public:
		//==========================================
		// 演算処理.
		// 入出力は処理依存
		//==========================================
		/** 演算処理を実行する.
			処理デバイス依存のメモリが渡される
			@param i_lppInputBuffer		入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@param o_lppOutputBuffer	出力データバッファ. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0が返る */
		virtual ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer) = 0;

		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する. 
			処理デバイス依存のメモリが渡される.
			@param	i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppOutputBuffer	出力データバッファ. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		virtual ErrorCode CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;

		/** 学習処理を実行する.
			処理デバイス依存のメモリが渡される.
			@param	i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppOutputBuffer	出力データバッファ. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		virtual ErrorCode Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif