//======================================
// バッチ正規化レイヤー
// CPU処理用
//======================================
#include"stdafx.h"

#include"BatchNormalization_DATA.hpp"
#include"BatchNormalization_FUNC.hpp"
#include"BatchNormalization_Base.h"

#include"BatchNormalization_CPU.h"
#include"BatchNormalization_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	BatchNormalization_CPU::BatchNormalization_CPU(Gravisbell::GUID guid, BatchNormalization_LayerData_CPU& i_layerData)
		:	BatchNormalization_Base	(guid)
		,	layerData				(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount		(0)				/**< 入力バッファ数 */
		,	outputBufferCount		(0)				/**< 出力バッファ数 */
		,	channeclBufferCount		(0)				/**< 1チャンネル当たりのバッファ数 */
		,	onLearnMode				(false)			/**< 学習処理中フラグ */
		,	learnCount				(0)				/**< 学習実行回数 */
	{
	}
	/** デストラクタ */
	BatchNormalization_CPU::~BatchNormalization_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 BatchNormalization_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode BatchNormalization_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	BatchNormalization_LayerData_Base& BatchNormalization_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const BatchNormalization_LayerData_Base& BatchNormalization_CPU::GetLayerData()const
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
	ErrorCode BatchNormalization_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 学習用の変数を作成
		this->onLearnMode = true;
		this->learnCount = 0;
		this->lpTmpMean.resize(this->layerData.inputDataStruct.ch, 0.0f);
		this->lpTmpVariance.resize(this->layerData.inputDataStruct.ch, 0.0f);

		// 出力誤差バッファ受け取り用のアドレス配列を作成する
		this->m_lppDOutputBufferPrev.resize(batchSize);

		// 入力誤差バッファ受け取り用のアドレス配列を作成する
		this->m_lppDInputBuffer.resize(batchSize);


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode BatchNormalization_CPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// チャンネルごとのバッファ数を確認
		this->channeclBufferCount = this->layerData.inputDataStruct.z * this->layerData.inputDataStruct.y * this->layerData.inputDataStruct.x;
		if(this->channeclBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 入力バッファ保存用のアドレス配列を作成
		this->m_lppInputBuffer.resize(batchSize, NULL);

		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum * this->outputBufferCount];
		}

		// 平均,分散を一時バッファに移す
		this->lpTmpMean = this->layerData.lpMean;
		this->lpTmpVariance = this->layerData.lpVariance;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode BatchNormalization_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		// 学習設定を保存
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		this->pLearnData->WriteToStruct((BYTE*)&learnData);


		// 学習回数を初期化
		this->learnCount = 0;

		// 演算用の平均.分散を初期化
		for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
		{
			this->layerData.lpMean[ch] = 0.0f;
			this->layerData.lpVariance[ch] = 0.0f;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode BatchNormalization_CPU::PreProcessCalculateLoop()
	{
		// 平均,分散を一時バッファに移す
		this->lpTmpMean = this->layerData.lpMean;
		this->lpTmpVariance = this->layerData.lpVariance;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode BatchNormalization_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];

		// 学習中ならば平均、分散を求める
		if(this->onLearnMode)
		{
			// 平均を求める
			for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
			{
				this->lpTmpMean[ch] = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
					{
						this->lpTmpMean[ch] += this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];
					}
				}
				this->lpTmpMean[ch] /= (this->batchSize * this->channeclBufferCount);
			}

			// 分散を求める
			for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
			{
				F64 tmp = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
					{
						F64 value = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];

						tmp += (value - this->lpTmpMean[ch]) * (value - this->lpTmpMean[ch]);
					}
				}
				this->lpTmpVariance[ch]    = (F32)(tmp / (this->batchSize * this->channeclBufferCount));
			}
		}

		// 平均,分散を利用して正規化
		for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
		{
			F32 mean = this->lpTmpMean[ch];
			F32 variance = this->lpTmpVariance[ch] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F32 sqrtVariance = (F32)sqrt(variance);

			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
				{
					F32 value = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];

					// 正規化
					F32 value2 = (value - mean) / sqrtVariance;

					// スケーリングとバイアス
					this->lppBatchOutputBuffer[batchNum][this->channeclBufferCount*ch + bufNum] = this->layerData.lpScale[ch] * value2 + this->layerData.lpBias[ch];
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER BatchNormalization_CPU::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode BatchNormalization_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseOutputBuffer = this->GetOutputBuffer();

		memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32) * outputBufferCount * batchSize);

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
	ErrorCode BatchNormalization_CPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 出力誤差バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppDOutputBufferPrev[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];

		// 入力誤差バッファのアドレスを配列に格納
		this->m_lpDInputBuffer = o_lppDInputBuffer;
		if(o_lppDInputBuffer)
		{
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				this->m_lppDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
		}


		for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
		{
			F32 mean = this->lpTmpMean[ch];
			F32 variance = this->lpTmpVariance[ch] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F32 sqrtVariance  = (F32)sqrt(variance);
			F32 sqrtVariance3 = sqrtVariance*sqrtVariance*sqrtVariance;
			F32 scale = this->layerData.lpScale[ch];

			// 平均と分散の誤差の合計値を求める
			F32 dMean = 0.0f;
			F32 dVariance = 0.0f;
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
				{
					F32 value  = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];
					F32 dOutput = this->m_lppDOutputBufferPrev[batchNum][this->channeclBufferCount*ch + bufNum];
					F32 dValue2 = dOutput * scale;

					dVariance += dValue2 * (value - mean) * (-1) / 2 / sqrtVariance3;
					dMean     += dValue2 * (-1) / sqrtVariance;
				}
			}

			// 入力誤差を求める
			if(o_lppDInputBuffer)
			{
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
					{
						F32 value  = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];
						F32 dOutput = this->m_lppDOutputBufferPrev[batchNum][this->channeclBufferCount*ch + bufNum];
						F32 dValue2 = dOutput * scale;

						this->m_lppDInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum]
							= dValue2 / sqrtVariance
							+ dVariance * 2 * (value - mean) / (this->channeclBufferCount * this->batchSize)
							+ dMean / (this->channeclBufferCount * this->batchSize);
					}
				}
			}
		}


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode BatchNormalization_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		Gravisbell::ErrorCode errCode = this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;


		// 平均値更新用の係数を算出
		F64 factor = 1.0 / (this->learnCount+1);

		// 学習処理の実行回数をカウントアップ
		this->learnCount++;

		for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
		{
			F32 mean = this->lpTmpMean[ch];
			F32 variance = this->lpTmpVariance[ch] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F64 sqrtVariance = (F32)sqrt(variance);
			F64 sqrtVarianceInv = 1.0f / sqrtVariance;

			F32 dBias = 0.0f;
			F32 dScale = 0.0f;

			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
				{
					F32 value = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];

					// 正規化
					F32 value2 = (F32)( (value - mean) * sqrtVarianceInv );

					dScale += this->m_lppDOutputBufferPrev[batchNum][this->channeclBufferCount*ch + bufNum] * value2;
					dBias  += this->m_lppDOutputBufferPrev[batchNum][this->channeclBufferCount*ch + bufNum];
				}
			}

			// 値を更新
			this->layerData.lpScale[ch] += this->learnData.LearnCoeff * dScale;
			this->layerData.lpBias[ch]  += this->learnData.LearnCoeff * dBias;

			// 平均と分散を更新
			this->layerData.lpMean[ch]     = (F32)((1.0 - factor) * this->layerData.lpMean[ch]     + factor * this->lpTmpMean[ch]);
			this->layerData.lpVariance[ch] = (F32)((1.0 - factor) * this->layerData.lpVariance[ch] + factor * variance);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER BatchNormalization_CPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer;
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode BatchNormalization_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseDInputBuffer = this->GetDInputBuffer();

		memcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32) * inputBufferCount * batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
