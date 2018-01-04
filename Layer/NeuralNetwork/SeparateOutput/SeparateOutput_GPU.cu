//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"SeparateOutput_DATA.hpp"
#include"SeparateOutput_FUNC.hpp"
#include"SeparateOutput_Base.h"

#include"SeparateOutput_GPU.cuh"
#include"SeparateOutput_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	SeparateOutput_GPU::SeparateOutput_GPU(Gravisbell::GUID guid, SeparateOutput_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	SeparateOutput_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	SeparateOutput_GPU::~SeparateOutput_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 SeparateOutput_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode SeparateOutput_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	SeparateOutput_LayerData_Base& SeparateOutput_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const SeparateOutput_LayerData_Base& SeparateOutput_GPU::GetLayerData()const
	{
		return this->layerData;
	}


	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode SeparateOutput_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode SeparateOutput_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;


		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode SeparateOutput_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode SeparateOutput_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力バッファを出力バッファにコピー
		cudaMemcpy(o_lppOutputBuffer, i_lppInputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列の[GetOutputDataCount()]配列
		直前の計算結果を使用する */
	ErrorCode SeparateOutput_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// バッファ1つ目の出力誤差で上書き
			cudaMemcpy(o_lppDInputBuffer, i_lppDOutputBuffer[0], sizeof(F32)*this->GetBatchSize()*this->outputBufferCount, cudaMemcpyDeviceToDevice);

			// 加算
			for(U32 outputLayerNum=1; outputLayerNum<(U32)this->layerData.layerStructure.separateCount; outputLayerNum++)
			{
				F32 alpha = 1.0f;

				cublasSaxpy_v2(
					this->cublasHandle,
					this->GetBatchSize()*this->outputBufferCount,
					&alpha,
					i_lppDOutputBuffer[outputLayerNum], 1,
					o_lppDInputBuffer, 1);
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode SeparateOutput_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
