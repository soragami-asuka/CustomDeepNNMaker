//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"Activation_DATA.hpp"
#include"Activation_FUNC.hpp"
#include"Activation_Base.h"

#include"Activation_CPU.h"
#include"Activation_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace
{
	//================================
	// 活性化関数
	//================================
	// lenear系
	F32 func_activation_lenear(F32 x)
	{
		return x;
	}
	F32 func_dactivation_lenear(F32 x)
	{
		return 1;
	}

	// sigmoid系
	F32 func_activation_sigmoid(F32 x)
	{
		return 1.0f / (1.0f + (F32)exp(-x));
	}
	F32 func_dactivation_sigmoid(F32 x)
	{
		return x * (1.0f - x);
	}

	F32 func_activation_sigmoid_crossEntropy(F32 x)
	{
		return 1.0f / (1.0f + (F32)exp(-x));
	}
	F32 func_dactivation_sigmoid_crossEntropy(F32 x)
	{
		return 1.0f;
	}

	// ReLU系
	F32 func_activation_ReLU(F32 x)
	{
		return x * (x > 0.0f);
	}
	F32 func_dactivation_ReLU(F32 x)
	{
		return 1.0f * (x > 0.0f);
	}

	// tanh系
	F32 func_activation_tanh(F32 x)
	{
		return tanh(x);
	}
	F32 func_dactivation_tanh(F32 x)
	{
		return 1.0f - x*x;
	}

	// SoftMax系
	F32 func_activation_SoftMax(F32 x)
	{
		return (F32)exp(x);	// 平均は別に行う
	}
}


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Activation_CPU::Activation_CPU(Gravisbell::GUID guid, Activation_LayerData_CPU& i_layerData)
		:	Activation_Base	(guid)
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
		,	func_activation					(func_activation_sigmoid)
		,	func_dactivation				(func_dactivation_sigmoid)
	{
	}
	/** デストラクタ */
	Activation_CPU::~Activation_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Activation_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Activation_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Activation_LayerData_Base& Activation_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const Activation_LayerData_Base& Activation_CPU::GetLayerData()const
	{
		return this->layerData;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 Activation_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		return this->layerData.WriteToBuffer(o_lpBuffer);
	}


	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode Activation_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 出力誤差バッファ受け取り用のアドレス配列を作成する
		this->m_lppDOutputBufferPrev.resize(batchSize);

		// 入力差分バッファを作成
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

		default:
			this->lpDInputBuffer.resize(this->batchSize * this->inputBufferCount);
			this->lppBatchDInputBuffer.resize(this->batchSize);
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum] = &this->lpDInputBuffer[batchNum * this->inputBufferCount];
			}
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Activation_CPU::PreProcessCalculate(unsigned int batchSize)
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

		// 入力バッファ保存用のアドレス配列を作成
		this->m_lppInputBuffer.resize(batchSize, NULL);

		// 出力バッファを作成
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

		default:
			this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);
			this->lppBatchOutputBuffer.resize(this->batchSize);
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum * this->outputBufferCount];
			}
			break;
		}


		// 活性化関数を設定
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			this->func_activation  = ::func_activation_lenear;
			this->func_dactivation = ::func_dactivation_lenear;
			break;

			// Sigmoid
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid:
		default:
			this->func_activation  = ::func_activation_sigmoid;
			this->func_dactivation = ::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid_crossEntropy:
			this->func_activation  = ::func_activation_sigmoid_crossEntropy;
			this->func_dactivation = ::func_dactivation_sigmoid_crossEntropy;
			break;

			// ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_ReLU:
			this->func_activation  = ::func_activation_ReLU;
			this->func_dactivation = ::func_dactivation_ReLU;
			break;

			// SoftMax系
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
			this->func_activation  = ::func_activation_SoftMax;
			this->func_dactivation = ::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			this->func_activation  = ::func_activation_SoftMax;
			this->func_dactivation = ::func_dactivation_sigmoid_crossEntropy;
			break;
		}

		// 演算用のバッファを確保
		switch(this->layerData.layerStructure.ActivationType)
		{
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			this->lpCalculateSum.resize(this->layerData.inputDataStruct.z * this->layerData.inputDataStruct.y * this->layerData.inputDataStruct.x);
			break;
		default:
			this->lpCalculateSum.clear();
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Activation_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Activation_CPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Activation_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファのアドレスを配列に格納
		this->m_lpInputBuffer = i_lpInputBuffer;
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];

		
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

		default:
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					// 活性化
					this->lppBatchOutputBuffer[batchNum][inputNum] = this->func_activation(this->m_lppInputBuffer[batchNum][inputNum]);
				}
			}
			break;
		}

		// softMaxを実行する
		switch(this->layerData.layerStructure.ActivationType)
		{
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
			{
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					// 合計値を算出
					F32 sum = 0.0f;
					for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
					{
						sum += this->lppBatchOutputBuffer[batchNum][outputNum];
					}

					// 値を合計値で割る
					for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
					{
						if(sum == 0.0f)
							this->lppBatchOutputBuffer[batchNum][outputNum] = 1.0f / this->outputBufferCount;
						else
							this->lppBatchOutputBuffer[batchNum][outputNum] /= sum;
					}
				}
			}
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			{
				U32 chSize = this->layerData.inputDataStruct.z * this->layerData.inputDataStruct.y * this->layerData.inputDataStruct.x;

				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					// 一時バッファクリア
					memset(&this->lpCalculateSum[0], 0, this->lpCalculateSum.size()*sizeof(F32));

					// 合計値を算出
					for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
					{
						for(U32 z=0; z<this->layerData.inputDataStruct.z; z++)
						{
							for(U32 y=0; y<this->layerData.inputDataStruct.y; y++)
							{
								for(U32 x=0; x<this->layerData.inputDataStruct.x; x++)
								{
									U32 offset = (((((ch*this->layerData.inputDataStruct.z+z)*this->layerData.inputDataStruct.y)+y)*this->layerData.inputDataStruct.x)+x);

									this->lpCalculateSum[offset] += this->lppBatchOutputBuffer[batchNum][ch*chSize + offset];
								}
							}
						}
					}

					// 合計値で割る
					for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
					{
						for(U32 z=0; z<this->layerData.inputDataStruct.z; z++)
						{
							for(U32 y=0; y<this->layerData.inputDataStruct.y; y++)
							{
								for(U32 x=0; x<this->layerData.inputDataStruct.x; x++)
								{
									U32 offset = (((((ch*this->layerData.inputDataStruct.z+z)*this->layerData.inputDataStruct.y)+y)*this->layerData.inputDataStruct.x)+x);

									this->lppBatchOutputBuffer[batchNum][ch*chSize + offset] /= this->lpCalculateSum[offset];
								}
							}
						}
					}
				}
			}
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Activation_CPU::GetOutputBuffer()const
	{
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			return this->m_lpInputBuffer;
			break;

		default:
			return &this->lpOutputBuffer[0];
		}
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode Activation_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// 学習処理
	//================================
	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Activation_CPU::Training(CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		// 出力誤差バッファのアドレスを配列に格納
		this->m_lpDOutputBufferPrev = i_lpDOutputBufferPrev;
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppDOutputBufferPrev[batchNum] = &i_lpDOutputBufferPrev[batchNum * this->outputBufferCount];


		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

		default:
			// 出力誤差を計算
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					this->lppBatchDInputBuffer[batchNum][inputNum] = this->func_dactivation(this->lppBatchOutputBuffer[batchNum][inputNum]) * this->m_lppDOutputBufferPrev[batchNum][inputNum];
				}
			}
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Activation_CPU::GetDInputBuffer()const
	{
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			return this->m_lpDOutputBufferPrev;
			break;

		default:
			return &this->lpDInputBuffer[0];
			break;
		}
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode Activation_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		memcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*inputBufferCount*batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
