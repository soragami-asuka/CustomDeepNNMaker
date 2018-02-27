//======================================
// バッチ正規化レイヤー
// GPU処理用
//======================================
#include"stdafx.h"

#include"Normalization_Scale_DATA.hpp"
#include"Normalization_Scale_FUNC.hpp"
#include"Normalization_Scale_Base.h"

#include"Normalization_Scale_GPU.cuh"
#include"Normalization_Scale_LayerData_GPU.cuh"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Normalization_Scale_GPU::Normalization_Scale_GPU(Gravisbell::GUID guid, Normalization_Scale_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Normalization_Scale_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData					(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount			(0)				/**< 入力バッファ数 */
		,	outputBufferCount			(0)				/**< 出力バッファ数 */
	{
	}
	/** デストラクタ */
	Normalization_Scale_GPU::~Normalization_Scale_GPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Normalization_Scale_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Normalization_Scale_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Normalization_Scale_LayerData_Base& Normalization_Scale_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Normalization_Scale_LayerData_Base& Normalization_Scale_GPU::GetLayerData()const
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
	ErrorCode Normalization_Scale_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 学習用の変数を作成

		// 入力誤差バッファ
		this->m_lpDInputBuffer_h.resize(this->GetBatchSize() * this->inputBufferCount);
		this->m_lppDInputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->m_lppDInputBuffer_h[i] = &this->m_lpDInputBuffer_h[i * this->inputBufferCount];

		// 出力誤差バッファ
		this->m_lpDOutputBuffer_h.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lppDOutputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->m_lppDOutputBuffer_h[i] = &this->m_lpDOutputBuffer_h[i * this->outputBufferCount];

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Normalization_Scale_GPU::PreProcessCalculate()
	{
		// 平均値用のバッファを作成
		this->lpTmpMean.resize(this->GetBatchSize());

		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 入力バッファを作成
		this->m_lpInputBuffer_h.resize(this->inputBufferCount * this->GetBatchSize());
		this->m_lppInputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->m_lppInputBuffer_h[i] = &this->m_lpInputBuffer_h[i * this->inputBufferCount];

		// 出力バッファを作成
		this->m_lpOutputBuffer_h.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lppOutputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->m_lppOutputBuffer_h[i] = &this->m_lpOutputBuffer_h[i * this->outputBufferCount];


		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Normalization_Scale_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Normalization_Scale_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力バッファをホストにコピー
		cudaMemcpy(&this->m_lpInputBuffer_h[0], i_lppInputBuffer, sizeof(F32)*this->m_lpInputBuffer_h.size(), cudaMemcpyDeviceToHost);

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			// 平均値を求める
			F32 ave = 0.0f;
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				ave += this->m_lppInputBuffer_h[batchNum][inputNum];
			}
			ave /= this->inputBufferCount;

			if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN)
				this->lpTmpMean[batchNum] = ave;

			// 出力を計算する
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				this->m_lppOutputBuffer_h[batchNum][inputNum] = (this->m_lppInputBuffer_h[batchNum][inputNum] - ave) * this->layerData.scale + this->layerData.bias;
			}
		}

		// 出力バッファをデバイスにコピー
		cudaMemcpy(o_lppOutputBuffer, &this->m_lpOutputBuffer_h[0], sizeof(F32)*this->m_lpOutputBuffer_h.size(), cudaMemcpyHostToDevice);


		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Normalization_Scale_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 出力誤差バッファをホストにコピー
		cudaMemcpy(&this->m_lpDOutputBuffer_h[0], i_lppDOutputBuffer, sizeof(F32)*this->m_lpDOutputBuffer_h.size(), cudaMemcpyDeviceToHost);

		// 入力誤差バッファのアドレスを配列に格納
		this->m_lpDInputBuffer_d = o_lppDInputBuffer;
		if(o_lppDInputBuffer)
		{
			// 入力誤差を計算
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					this->m_lppDInputBuffer_h[batchNum][inputNum] = this->layerData.scale * (1.0f - 1.0f/this->inputBufferCount) * this->m_lppDOutputBuffer_h[batchNum][inputNum];
				}
			}

			// 入力誤差をデバイスにコピー
			cudaMemcpy(o_lppDInputBuffer, &this->m_lpDInputBuffer_h[0], sizeof(F32)*this->m_lpDInputBuffer_h.size(), cudaMemcpyHostToDevice);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Normalization_Scale_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// スケールとバイアスの変化量を計算
		F32 dScale = 0.0f;
		F32 dBias  = 0.0f;

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			F32 ave = this->lpTmpMean[batchNum];

			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				dScale += this->m_lppDOutputBuffer_h[batchNum][inputNum] * (this->m_lppInputBuffer_h[batchNum][inputNum] - ave);
				dBias  += this->m_lppDOutputBuffer_h[batchNum][inputNum];
			}
		}

		// スケールとバイアスを更新
		if(this->layerData.m_pOptimizer_scale)
			this->layerData.m_pOptimizer_scale->UpdateParameter(&this->layerData.scale, &dScale);
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(&this->layerData.bias, &dBias);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
