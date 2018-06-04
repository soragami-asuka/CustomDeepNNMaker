//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"FullyConnect_DATA.hpp"
#include"FullyConnect_FUNC.hpp"
#include"FullyConnect_Base.h"

#include"FullyConnect_CPU.h"
#include"FullyConnect_LayerData_CPU.h"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	FullyConnect_CPU::FullyConnect_CPU(Gravisbell::GUID guid, FullyConnect_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FullyConnect_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	neuronCount						(0)		/**< ニューロン数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
	{
	}
	/** デストラクタ */
	FullyConnect_CPU::~FullyConnect_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 FullyConnect_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode FullyConnect_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	FullyConnect_LayerData_Base& FullyConnect_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const FullyConnect_LayerData_Base& FullyConnect_CPU::GetLayerData()const
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
	ErrorCode FullyConnect_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 入力誤差/出力誤差バッファ受け取り用のアドレス配列を作成する
		this->m_lppDInputBuffer.resize(this->GetBatchSize(), NULL);
		this->m_lppDOutputBuffer.resize(this->GetBatchSize(), NULL);


		// パラメータの変化量バッファ
		this->lpDBias.resize(this->neuronCount);
		this->lpDNeuron.resize(this->neuronCount * this->inputBufferCount);


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FullyConnect_CPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// ニューロン数を確認
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 入力/出力バッファ保存用のアドレス配列を作成
		this->m_lppInputBuffer.resize(this->GetBatchSize(), NULL);
		this->m_lppOutputBuffer.resize(this->GetBatchSize(), NULL);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FullyConnect_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FullyConnect_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->m_lppOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN && this->GetRuntimeParameterByStructure().UpdateWeigthWithOutputVariance)
		{
			U32 PROCTIME_MAX = 5;			// 実行最大値
			F32	VARIANCE_TOLERANCE = 0.1f;	// 分散交差(許容範囲)

			// バッファを確保
			std::vector<F32> lpTmpWeight(this->layerData.pWeightData->GetWeigthSize());
			std::vector<F32> lpTmpBias(this->layerData.pWeightData->GetBiasSize());

			// バッファをコピー
			memcpy(&lpTmpWeight[0], this->layerData.pWeightData->GetWeight(), sizeof(F32)*lpTmpWeight.size());
			memcpy(&lpTmpBias[0],   this->layerData.pWeightData->GetBias(),   sizeof(F32)*lpTmpBias.size());

			U32 procTime = 0;
			do
			{
				// 演算を実行
				ErrorCode err = this->CalculateBase(&lpTmpWeight[0], &lpTmpBias[0]);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				// 出力の分散を求める
				F32 variance = 0.0f;
				F32 average  = 0.0f;
				{
					// 平均を求める
					for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					{
						for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
						{
							average += this->m_lppOutputBuffer[batchNum][outputNum];
						}
					}
					average /= (this->outputBufferCount * this->GetBatchSize());

					// 分散を求める
					for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					{
						for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
						{
							variance += (this->m_lppOutputBuffer[batchNum][outputNum] - average) * (this->m_lppOutputBuffer[batchNum][outputNum] - average);
						}
					}
					variance /= (this->outputBufferCount * this->GetBatchSize());
				}

				if( abs(variance - 1.0f) < VARIANCE_TOLERANCE)
					break;

				// 標準偏差で重みを割って更新する
				F32 deviation = sqrtf(variance);
				{
					for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
						{
							lpTmpWeight[neuronNum*this->inputBufferCount + inputNum] /= deviation;
						}
						lpTmpBias[neuronNum] /= deviation;
					}
				}

				procTime++;
			}while(procTime < 5);

			// 重みを更新
			this->layerData.pWeightData->SetData(&lpTmpWeight[0], &lpTmpBias[0]);
		}
		else
		{
			ErrorCode err = this->CalculateBase(this->layerData.pWeightData->GetWeight(), this->layerData.pWeightData->GetBias());
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する. */
	ErrorCode FullyConnect_CPU::CalculateBase(const F32* lpWeight, const F32* lpBias)
	{
		for(unsigned int batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				float tmp = 0;

				// ニューロンの値を加算
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					tmp += this->m_lppInputBuffer[batchNum][inputNum] * lpWeight[neuronNum*inputBufferCount + inputNum];
				}
				tmp += lpBias[neuronNum];

				// 格納
				this->m_lppOutputBuffer[batchNum][neuronNum] = tmp;

#ifdef _DEBUG
				if(isnan(this->m_lppOutputBuffer[batchNum][neuronNum]))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
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
	ErrorCode FullyConnect_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum]   = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->m_lppOutputBuffer[batchNum]  = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
			this->m_lppDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
		}

		
		// 入力誤差差分を計算
		if(o_lppDInputBuffer)
		{
			// 入力誤差バッファのアドレスを配列に格納
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->m_lppDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					float tmp = 0.0f;

					for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						tmp += this->m_lppDOutputBuffer[batchNum][neuronNum] * this->layerData.pWeightData->GetWeight()[neuronNum*inputBufferCount + inputNum];
					}

					this->m_lppDInputBuffer[batchNum][inputNum] = tmp;
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FullyConnect_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// パラメータ変化量のバッファをクリア
		memset(&this->lpDBias[0],   0, sizeof(F32)*this->lpDBias.size());
		memset(&this->lpDNeuron[0], 0, sizeof(F32)*this->lpDNeuron.size());

		// 学習誤差を計算
		for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// バイアス更新
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					 sumDOutput += this->m_lppDOutputBuffer[batchNum][neuronNum];
				}

				this->lpDBias[neuronNum] = sumDOutput;
			}

			// 入力対応ニューロン更新
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					sumDOutput += this->m_lppInputBuffer[batchNum][inputNum] * this->m_lppDOutputBuffer[batchNum][neuronNum];
				}

				this->lpDNeuron[neuronNum*this->inputBufferCount + inputNum] += sumDOutput;
			}
		}

		// 誤差を反映
		this->layerData.pWeightData->UpdateData(&this->lpDNeuron[0], &this->lpDBias[0]);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
