//======================================
// バッチ正規化レイヤー
// CPU処理用
//======================================
#include"stdafx.h"

#include"BatchNormalizationAll_DATA.hpp"
#include"BatchNormalizationAll_FUNC.hpp"
#include"BatchNormalizationAll_Base.h"

#include"BatchNormalizationAll_CPU.h"
#include"BatchNormalizationAll_LayerData_CPU.h"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	BatchNormalizationAll_CPU::BatchNormalizationAll_CPU(Gravisbell::GUID guid, BatchNormalizationAll_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	BatchNormalizationAll_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount		(0)				/**< 入力バッファ数 */
		,	outputBufferCount		(0)				/**< 出力バッファ数 */
		,	onLearnMode				(false)			/**< 学習処理中フラグ */
		,	learnCount				(0)				/**< 学習実行回数 */
	{
	}
	/** デストラクタ */
	BatchNormalizationAll_CPU::~BatchNormalizationAll_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 BatchNormalizationAll_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode BatchNormalizationAll_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	ILayerData& BatchNormalizationAll_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& BatchNormalizationAll_CPU::GetLayerData()const
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
	ErrorCode BatchNormalizationAll_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 学習用の変数を作成
		this->onLearnMode = true;
		this->learnCount = 0;
		this->lpTmpMean.resize(this->layerData.lpMean.size(), 0.0f);
		this->lpTmpVariance.resize(this->layerData.lpVariance.size(), 0.0f);

		// 入力誤差/出力誤差バッファ受け取り用のアドレス配列を作成する
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());
		this->lppBatchDInputBuffer.resize(this->GetBatchSize());

		// パラメータ変化量のバッファを確保
		this->lpDBias.resize(this->layerData.lpBias.size());
		this->lpDScale.resize(this->layerData.lpScale.size());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode BatchNormalizationAll_CPU::PreProcessCalculate()
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

		// 平均,分散を一時バッファに移す
		this->lpTmpMean = this->layerData.lpMean;
		this->lpTmpVariance = this->layerData.lpVariance;

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode BatchNormalizationAll_CPU::PreProcessLoop()
	{
		switch(this->GetProcessType())
		{
		case ProcessType::PROCESSTYPE_LEARN:
			{
				// 学習回数を初期化
				this->learnCount = 0;

				// 演算用の平均.分散を初期化
				for(U32 ch=0; ch<1; ch++)
				{
					this->layerData.lpMean[ch] = 0.0f;
					this->layerData.lpVariance[ch] = 0.0f;
				}
			}
			break;
		case ProcessType::PROCESSTYPE_CALCULATE:	
			{
				// 平均,分散を一時バッファに移す
				this->lpTmpMean = this->layerData.lpMean;
				this->lpTmpVariance = this->layerData.lpVariance;
			}
			break;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode BatchNormalizationAll_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		// 学習中ならば平均、分散を求める
		if(this->onLearnMode)
		{
			// 平均を求める
			{
				this->lpTmpMean[0] = 0.0f;
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
					{
						this->lpTmpMean[0] += this->lppBatchInputBuffer[batchNum][bufNum];
					}
				}
				this->lpTmpMean[0] /= (this->GetBatchSize() * this->outputBufferCount);
			}

			// 分散を求める
			{
				F64 tmp = 0.0f;
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
					{
						F64 value = this->lppBatchInputBuffer[batchNum][bufNum];

						tmp += (value - this->lpTmpMean[0]) * (value - this->lpTmpMean[0]);
					}
				}
				this->lpTmpVariance[0] = (F32)(tmp / (this->GetBatchSize() * this->outputBufferCount));
			}
		}

		// 平均,分散を利用して正規化
		{
			F32 mean = this->lpTmpMean[0];
			F32 variance = this->lpTmpVariance[0] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F32 sqrtVariance = (F32)sqrt(variance);

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
				{
					F32 value = this->lppBatchInputBuffer[batchNum][bufNum];

					// 正規化
					F32 value2 = (value - mean) / sqrtVariance;

					// スケーリングとバイアス
					this->lppBatchOutputBuffer[batchNum][bufNum] = this->layerData.lpScale[0] * value2 + this->layerData.lpBias[0];
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
	ErrorCode BatchNormalizationAll_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
		}

		// 入力誤差バッファのアドレスを配列に格納配列に格納
		if(o_lppDInputBuffer)
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
			}
		}

		// 出力誤差バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
		}


		{
			F32 mean = this->lpTmpMean[0];
			F32 variance = this->lpTmpVariance[0] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F32 sqrtVariance  = (F32)sqrt(variance);
			F32 sqrtVariance3 = sqrtVariance*sqrtVariance*sqrtVariance;
			F32 scale = this->layerData.lpScale[0];

			// 平均と分散の誤差の合計値を求める
			F32 dMean = 0.0f;
			F32 dVariance = 0.0f;
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
				{
					F32 value  = this->lppBatchInputBuffer[batchNum][bufNum];
					F32 dOutput = this->lppBatchDOutputBuffer[batchNum][bufNum];
					F32 dValue2 = dOutput * scale;

					dVariance += dValue2 * (value - mean) * (-1) / 2 / sqrtVariance3;
					dMean     += dValue2 * (-1) / sqrtVariance;
				}
			}

			// 入力誤差を求める
			if(o_lppDInputBuffer)
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
					{
						F32 value  = this->lppBatchInputBuffer[batchNum][bufNum];
						F32 dOutput = this->lppBatchDOutputBuffer[batchNum][bufNum];
						F32 dValue2 = dOutput * scale;

						this->lppBatchDInputBuffer[batchNum][bufNum]
							= dValue2 / sqrtVariance
							+ dVariance * 2 * (value - mean) / (this->outputBufferCount * this->GetBatchSize())
							+ dMean / (this->outputBufferCount * this->GetBatchSize());
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
	ErrorCode BatchNormalizationAll_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		Gravisbell::ErrorCode errCode = this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;


		// 平均値更新用の係数を算出
		F64 factor = max(1.0 / (this->learnCount+1), this->GetRuntimeParameterByStructure().AverageUpdateCoeffMin);

		// 学習処理の実行回数をカウントアップ
		this->learnCount++;

		{
			F32 mean = this->lpTmpMean[0];
			F32 variance = this->lpTmpVariance[0] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F64 sqrtVariance = (F32)sqrt(variance);
			F64 sqrtVarianceInv = 1.0f / sqrtVariance;

			this->lpDScale[0] = 0.0f;
			this->lpDBias[0]  = 0.0f;

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
				{
					F32 value = this->lppBatchInputBuffer[batchNum][bufNum];

					// 正規化
					F32 value2 = (F32)( (value - mean) * sqrtVarianceInv );

					this->lpDScale[0] += this->lppBatchDOutputBuffer[batchNum][bufNum] * value2;
					this->lpDBias[0]  += this->lppBatchDOutputBuffer[batchNum][bufNum];
				}
			}

			// 平均と分散を更新
			this->layerData.lpMean[0]     = (F32)((1.0 - factor) * this->layerData.lpMean[0]     + factor * this->lpTmpMean[0]);
			this->layerData.lpVariance[0] = (F32)((1.0 - factor) * this->layerData.lpVariance[0] + factor * variance);
		}

		// スケールとバイアスを更新
		if(this->layerData.m_pOptimizer_scale)
			this->layerData.m_pOptimizer_scale->UpdateParameter(&this->layerData.lpScale[0], &this->lpDScale[0]);
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(&this->layerData.lpBias[0], &this->lpDBias[0]);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
