//======================================
// バッチ正規化レイヤー
// CPU処理用
//======================================
#include"stdafx.h"

#include"Normalization_Scale_DATA.hpp"
#include"Normalization_Scale_FUNC.hpp"
#include"Normalization_Scale_Base.h"

#include"Normalization_Scale_CPU.h"
#include"Normalization_Scale_LayerData_CPU.h"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Normalization_Scale_CPU::Normalization_Scale_CPU(Gravisbell::GUID guid, Normalization_Scale_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Normalization_Scale_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount		(0)				/**< 入力バッファ数 */
		,	outputBufferCount		(0)				/**< 出力バッファ数 */
	{
	}
	/** デストラクタ */
	Normalization_Scale_CPU::~Normalization_Scale_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Normalization_Scale_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Normalization_Scale_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	ILayerData& Normalization_Scale_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& Normalization_Scale_CPU::GetLayerData()const
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
	ErrorCode Normalization_Scale_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 学習用の変数を作成
		this->lpTmpMean.resize(this->GetBatchSize(), 0.0f);

		// 入力誤差/出力誤差バッファ受け取り用のアドレス配列を作成する
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());
		this->lppBatchDInputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Normalization_Scale_CPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 入力/出力バッファ保存用のアドレス配列を作成
		this->lppBatchInputBuffer.resize(this->GetBatchSize(), NULL);
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Normalization_Scale_CPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Normalization_Scale_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			// 平均値を求める
			F32 ave = 0.0f;
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				ave += this->lppBatchInputBuffer[batchNum][inputNum];
			}
			ave /= this->inputBufferCount;

			if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN)
				this->lpTmpMean[batchNum] = ave;

			// 出力を計算する
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				this->lppBatchOutputBuffer[batchNum][inputNum] = (this->lppBatchInputBuffer[batchNum][inputNum] - ave) * this->layerData.scale + this->layerData.bias;
			}
		}

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
	ErrorCode Normalization_Scale_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			// 入力誤差/出力誤差バッファのアドレスを配列に格納
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
				this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
			}

			// 入力誤差を計算
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					this->lppBatchDInputBuffer[batchNum][inputNum] = this->layerData.scale * (1.0f - 1.0f/this->inputBufferCount) * this->lppBatchDOutputBuffer[batchNum][inputNum];
				}
			}
		}


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Normalization_Scale_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// 入力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
		}

		// スケールとバイアスの変化量を計算
		F32 dScale = 0.0f;
		F32 dBias  = 0.0f;

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			F32 ave = this->lpTmpMean[batchNum];

			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				dScale += this->lppBatchDOutputBuffer[batchNum][inputNum] * (this->lppBatchInputBuffer[batchNum][inputNum] - ave);
				dBias  += this->lppBatchDOutputBuffer[batchNum][inputNum];
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
