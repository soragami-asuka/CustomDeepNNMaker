//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
// GPU処理(デバイスメモリ版)
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_BASE_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_BASE_H__

#include"FeedforwardNeuralNetwork_Base.h"

#include<Layer/NeuralNetwork/ILayerDLLManager.h>

#include<thrust/device_vector.h>

#include"../_LayerBase/CLayerBase_GPU.cuh"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_GPU_base : public FeedforwardNeuralNetwork_Base
	{
	private:
		std::vector<F32> lpOutputBuffer_h;	/**< 出力バッファ(ホストメモリ) */
		std::vector<thrust::device_vector<F32>> lpDInputBuffer;

		//====================================
		// コンストラクタ/デストラクタ
		//====================================
	public:
		/** コンストラクタ */
		FeedforwardNeuralNetwork_GPU_base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct);
		/** コンストラクタ */
		FeedforwardNeuralNetwork_GPU_base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
		/** デストラクタ */
		virtual ~FeedforwardNeuralNetwork_GPU_base();


		//====================================
		// 入力誤差バッファ関連
		//====================================
	protected:
		/** 入力誤差バッファの総数を設定する */
		ErrorCode SetDInputBufferCount(U32 i_DInputBufferCount);

		/** 入力誤差バッファのサイズを設定する */
		ErrorCode ResizeDInputBuffer(U32 i_DInputBufferNo, U32 i_bufferSize);

	public:
		/** 入力誤差バッファを取得する */
		BATCH_BUFFER_POINTER GetDInputBuffer_d(U32 i_DInputBufferNo);
		/** 入力誤差バッファを取得する */
		CONST_BATCH_BUFFER_POINTER GetDInputBuffer_d(U32 i_DInputBufferNo)const;


		//====================================
		// 出力バッファ関連
		//====================================
	protected:
		/** 出力バッファの総数を設定する */
		virtual ErrorCode SetOutputBufferCount(U32 i_outputBufferCount) = 0;

		/** 出力バッファのサイズを設定する */
		virtual ErrorCode ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize) = 0;

	public:
		/** 出力バッファの現在の使用者を取得する */
		virtual GUID GetReservedOutputBufferID(U32 i_outputBufferNo) = 0;
		/** 出力バッファを使用中にして取得する(処理デバイス依存) */
		virtual BATCH_BUFFER_POINTER ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid) = 0;


	public:
		//====================================
		// 入出力バッファ関連
		//====================================
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
		// 演算処理
		//================================
		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		ErrorCode PreProcessLearn(U32 batchSize);
		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		ErrorCode PreProcessCalculate(unsigned int batchSize);


		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);

		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);
		/** 演算処理を実行する.
			@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
			@return 成功した場合0が返る */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer);

		//================================
		// 学習処理
		//================================
		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列
			直前の計算結果を使用する */
		ErrorCode CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
		/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列
			直前の計算結果を使用する */
		ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
		/** 学習誤差を計算する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif