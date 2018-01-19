//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"SOM_DATA.hpp"
#include"SOM_FUNC.hpp"
#include"SOM_Base.h"

#include"SOM_CPU.h"
#include"SOM_LayerData_CPU.h"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {



	/** コンストラクタ */
	SOM_CPU::SOM_CPU(Gravisbell::GUID guid, SOM_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	SOM_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount		(0)		/**< 入力バッファ数 */
		,	unitCount				(0)		/**< ニューロン数 */
		,	outputBufferCount		(0)		/**< 出力バッファ数 */
		,	temporaryMemoryManager	(i_temporaryMemoryManager)
	{
	}
	/** デストラクタ */
	SOM_CPU::~SOM_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 SOM_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode SOM_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	SOM_LayerData_Base& SOM_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const SOM_LayerData_Base& SOM_CPU::GetLayerData()const
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
	ErrorCode SOM_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 入力誤差/出力誤差バッファ受け取り用のアドレス配列を作成する

		// パラメータの変化量バッファ
		this->lpDUnit.resize(this->unitCount * this->inputBufferCount);
		this->lppDUnit.resize(this->unitCount);
		for(U32 i=0; i<this->lppDUnit.size(); i++)
			this->lppDUnit[i] = (F32*)&this->lpDUnit[i * this->inputBufferCount];


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode SOM_CPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// ニューロン数を確認
		this->unitCount = this->GetUnitCount();
		if(this->unitCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ニューロンバッファのサイズ確認
		if(this->layerData.lpUnitData.size() != this->unitCount * this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->layerData.lppUnitData.size() != this->unitCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 入力/出力バッファ保存用のアドレス配列を作成
		this->m_lppInputBuffer.resize(this->GetBatchSize(), NULL);
		this->m_lppOutputBuffer.resize(this->GetBatchSize(), NULL);

		// 各ユニットの座標を計算する
		this->lpUnitPos.resize(this->unitCount);
		for(U32 unitNo=0; unitNo<this->unitCount; unitNo++)
		{
			this->lpUnitPos[unitNo].resize(this->layerData.layerStructure.DimensionCount);

			U32 tmpNo = unitNo;
			for(U32 dimNo=0; dimNo<(U32)this->layerData.layerStructure.DimensionCount; dimNo++)
			{
				U32 pos = tmpNo % this->layerData.layerStructure.ResolutionCount;
				tmpNo /= this->layerData.layerStructure.ResolutionCount;

				this->lpUnitPos[unitNo][dimNo] = (F32)pos / (this->layerData.layerStructure.ResolutionCount - 1);
			}
		}

		// 一時バッファを確保


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode SOM_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode SOM_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->m_lppOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		// BMU(Best Matching Unit)を調べる
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				S32 bmuNo = -1;
				F32 bmuMatchRate = 0.0f;

				for(U32 unitNo=0; unitNo<this->unitCount; unitNo++)
				{
					// 入力信号とユニットの内積を一致率とする(本当にあってる？)
					F32 matchRate=0;
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
					{
						matchRate += this->m_lppInputBuffer[batchNum][inputNum] * this->layerData.lppUnitData[unitNo][inputNum];
					}

					if(bmuNo<0 || matchRate>bmuMatchRate)
					{
						bmuNo = unitNo;
						bmuMatchRate = matchRate;
					}
				}

				// BMU番号からN次元座標に変換
				for(U32 dimNo=0; dimNo<(U32)this->layerData.layerStructure.DimensionCount; dimNo++)
					this->m_lppOutputBuffer[batchNum][dimNo] = this->lpUnitPos[bmuNo][dimNo];
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
	ErrorCode SOM_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			// 入力誤差バッファをクリア
			memset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode SOM_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum]   = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->m_lppOutputBuffer[batchNum]  = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
		}

		// 誤差を計算
		{
			// 時間減衰率を計算
			F32 timeAttenuationRate = this->GetRuntimeParameterByStructure().SOM_L0 * exp(-(S32)this->layerData.learnTime / this->GetRuntimeParameterByStructure().SOM_ramda);
			// 距離減衰率計算量の係数を計算
			F32 lengthAttenuationRateCoeff = 2.0f * pow(this->GetRuntimeParameterByStructure().SOM_sigma * exp(-(S32)this->layerData.learnTime / this->GetRuntimeParameterByStructure().SOM_ramda), 2);

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				// ユニットごとに誤差を計算
				for(U32 unitNo=0; unitNo<this->layerData.lppUnitData.size(); unitNo++)
				{
					// BMUとの距離の2乗を求める
					F32 length2 = 0.0f;
					for(U32 dimNo=0; dimNo<(U32)this->layerData.layerStructure.DimensionCount; dimNo++)
					{
						length2 += pow(this->m_lppOutputBuffer[batchNum][dimNo] - this->lpUnitPos[unitNo][dimNo], 2);
					}

					// 距離減衰率を求める
					F32 lengthAttenuationRate = exp(-length2 / lengthAttenuationRateCoeff);

					// 減衰率を求める
					F32 attenuationRate = timeAttenuationRate * lengthAttenuationRate;

					// 誤差の更新
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
					{
						F32 dValue = this->m_lppInputBuffer[batchNum][inputNum] - this->layerData.lppUnitData[unitNo][inputNum];

						this->lppDUnit[unitNo][inputNum] += attenuationRate * dValue;
					}
				}
			}
		}

		// 誤差をユニットに加算
		for(int i=0; i<this->layerData.lpUnitData.size(); i++)
		{
			this->layerData.lpUnitData[i] += this->lpDUnit[i] / this->GetBatchSize();
		}

		// 学習回数をカウントアップ
		this->layerData.learnTime++;

		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
