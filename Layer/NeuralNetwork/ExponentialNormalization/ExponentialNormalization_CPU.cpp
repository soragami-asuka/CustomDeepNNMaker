//======================================
// バッチ正規化レイヤー
// CPU処理用
//======================================
#include"stdafx.h"

#include<algorithm>

#include"ExponentialNormalization_DATA.hpp"
#include"ExponentialNormalization_FUNC.hpp"
#include"ExponentialNormalization_Base.h"

#include"ExponentialNormalization_CPU.h"
#include"ExponentialNormalization_LayerData_CPU.h"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	ExponentialNormalization_CPU::ExponentialNormalization_CPU(Gravisbell::GUID guid, ExponentialNormalization_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	ExponentialNormalization_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount		(0)				/**< 入力バッファ数 */
		,	outputBufferCount		(0)				/**< 出力バッファ数 */
		,	channeclBufferCount		(0)				/**< 1チャンネル当たりのバッファ数 */
		,	temporaryMemoryManager	(i_temporaryMemoryManager)	/**< 一時バッファ管理 */
	{
	}
	/** デストラクタ */
	ExponentialNormalization_CPU::~ExponentialNormalization_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 ExponentialNormalization_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode ExponentialNormalization_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	ILayerData& ExponentialNormalization_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& ExponentialNormalization_CPU::GetLayerData()const
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
	ErrorCode ExponentialNormalization_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 入力誤差/出力誤差バッファ受け取り用のアドレス配列を作成する
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());
		this->lppBatchDInputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode ExponentialNormalization_CPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// チャンネルごとのバッファ数を確認
		this->channeclBufferCount = this->GetInputDataStruct().z * this->GetInputDataStruct().y * this->GetInputDataStruct().x;
		if(this->channeclBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 入力/出力バッファ保存用のアドレス配列を作成
		this->lppBatchInputBuffer.resize(this->GetBatchSize(), NULL);
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode ExponentialNormalization_CPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode ExponentialNormalization_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		// 平均/分散を利用して正規化
		for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
		{
			F32 mean = this->layerData.lpMean[ch];
			F32 variance = this->layerData.lpVariance[ch] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F32 sqrtVariance = (F32)sqrt(variance);

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
				{
					F32 value = this->lppBatchInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];

					// 正規化
					this->lppBatchOutputBuffer[batchNum][this->channeclBufferCount*ch + bufNum] = (value - mean) / sqrtVariance;
				}
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
	ErrorCode ExponentialNormalization_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力/出力/出力誤差バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
			this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
		}

		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 入力誤差バッファのアドレスを配列に格納配列に格納
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
			}

			for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
			{
				F32 mean = this->layerData.lpMean[ch];
				F32 variance = this->layerData.lpVariance[ch] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
				F32 sqrtVariance = (F32)sqrt(variance);

				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
					{
						this->lppBatchDInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum]
							= this->lppBatchDOutputBuffer[batchNum][this->channeclBufferCount*ch + bufNum] / sqrtVariance;
					}
				}
			}

		}


#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		memcpy(&lpTmpInputBuffer[0], i_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size());

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		memcpy(&lpTmpOutputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size());

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		memcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size());

		std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		memcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size());
#endif


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode ExponentialNormalization_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// 学習回数を更新
		this->layerData.learnTime++;

		// CHごとに平均と分散を求める
		F32 alpha = 0.0f;
		if(this->layerData.learnTime < this->layerData.layerStructure.InitParameterTime)
			alpha = 1.0f / (this->layerData.learnTime + 1);
		else
			alpha = std::min<F32>(1.0f, this->GetRuntimeParameterByStructure().AccelCoeff * 2 / (this->layerData.layerStructure.ExponentialTime + 1));

		for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
		{
			// 平均を求める
			F32 average = 0.0f;
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
				{
					F32 value = this->lppBatchInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];

					average += value;
				}
			}
			average /= (this->GetBatchSize() * this->channeclBufferCount);

			// 平均を更新する
			this->layerData.lpMean[ch] = alpha * average + (1.0f - alpha) * this->layerData.lpMean[ch];

			// 分散を求める
			average = this->layerData.lpMean[ch];
			F32 variance = 0.0f;
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
				{
					F32 value = this->lppBatchInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];

					variance += (value - average) * (value - average);
				}
			}
			variance /= (this->GetBatchSize() * this->channeclBufferCount);

			// 分散を更新する
			this->layerData.lpVariance[ch] = alpha * variance + (1.0f - alpha) * this->layerData.lpVariance[ch];
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
