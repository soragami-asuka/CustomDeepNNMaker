//======================================
// 畳み込みニューラルネットワークの結合処理レイヤー
// GPU処理用
//======================================
#ifndef __CONVOLUTION_GPU_H__
#define __CONVOLUTION_GPU_H__

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


#include"Convolution_DATA.hpp"
#include"Convolution_FUNC.hpp"
#include"Convolution_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class Convolution_GPU : public Convolution_Base
{
private:
	// データ本体
	class Convolution_LayerData_GPU& layerData;

	// 入出力バッファ
	thrust::device_vector<F32>			lpOutputBuffer;		/**< 出力バッファ <バッチ数><畳み込み数> */

	// Get関数を使うと処理不可がかさむので一時保存用. PreCalculateで値を格納.
	U32 filterSize;						/**< フィルタサイズ */
	U32 inputBufferCount;				/**< 入力バッファ数 */
	U32 neuronCount;					/**< ニューロン数 */
	U32 outputBufferCount;				/**< 出力バッファ数 */

	// 演算時の入力データ
	CONST_BATCH_BUFFER_POINTER				m_lppInputBuffer_d;			/**< 演算時の入力データ */
	CONST_BATCH_BUFFER_POINTER				m_lppDOutputBuffer_d;		/**< 入力誤差計算時の出力誤差データ */
	BATCH_BUFFER_POINTER					m_lpDInputBuffer_d;			/**< 入力誤差バッファ */

	// 演算処理用のバッファ
	cudnnHandle_t cudnnHandle;

	// CUDNN用データ構造定義
	cudnnTensorDescriptor_t			inputTensorDesc;			/**< 入力データ構造 */
	cudnnTensorDescriptor_t			outputTensorDesc;			/**< 出力データ構造 */
	cudnnTensorDescriptor_t			biasTensorDesc;				/**< バイアスデータ構造 */
	cudnnFilterDescriptor_t			filterDesc;					/**< フィルター構造 */
	cudnnConvolutionDescriptor_t	convDesc;					/**< 畳み込み設定 */
	cudnnConvolutionFwdAlgo_t		useForwardAlgorithm;		/**< 前方伝播時に使用するアルゴリズム番号 */
	cudnnConvolutionBwdDataAlgo_t	useBackwardDataAlgorithm;	/**< 後方伝播時のデータ計算に使用するアルゴリズム番号 */
	cudnnConvolutionBwdFilterAlgo_t	useBackwardFilterAlgorithm;	/**< 後方伝播時のフィルタ計算に使用するアルゴリズム番号 */
	thrust::device_vector<BYTE>		workSpace;					/**< 処理用のメモリ.前方伝播、後方伝播全てで共用する. */


public:
	/** コンストラクタ */
	Convolution_GPU(Gravisbell::GUID guid, class Convolution_LayerData_GPU& i_layerData);
	/** デストラクタ */
	virtual ~Convolution_GPU();


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
	Convolution_LayerData_Base& GetLayerData();
	const Convolution_LayerData_Base& GetLayerData()const;


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
	ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer);

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
	ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const;
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const;

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif