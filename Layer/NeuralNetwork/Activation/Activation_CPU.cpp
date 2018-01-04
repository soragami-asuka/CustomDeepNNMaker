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



namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Activation_CPU::Activation_CPU(Gravisbell::GUID guid, Activation_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Activation_Base					(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
		,	func_activation					(&Activation_CPU::func_activation_sigmoid)
		,	func_dactivation				(&Activation_CPU::func_dactivation_sigmoid)
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
	ILayerData& Activation_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& Activation_CPU::GetLayerData()const
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
	ErrorCode Activation_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 入力誤差/出力誤差バッファ受け取り用のアドレス配列を作成する
		this->m_lppDInputBuffer.resize(this->GetBatchSize(), NULL);
		this->m_lppDOutputBuffer.resize(this->GetBatchSize(), NULL);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Activation_CPU::PreProcessCalculate()
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
		this->m_lppInputBuffer.resize(this->GetBatchSize(), NULL);
		this->m_lppOutputBuffer.resize(this->GetBatchSize(), NULL);


		// 活性化関数を設定
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			this->func_activation  = &Activation_CPU::func_activation_lenear;
			this->func_dactivation = &Activation_CPU::func_dactivation_lenear;
			break;

			// Sigmoid
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid:
		default:
			this->func_activation  = &Activation_CPU::func_activation_sigmoid;
			this->func_dactivation = &Activation_CPU::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid_crossEntropy:
			this->func_activation  = &Activation_CPU::func_activation_sigmoid_crossEntropy;
			this->func_dactivation = &Activation_CPU::func_dactivation_sigmoid_crossEntropy;
			break;

			// ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_ReLU:
			this->func_activation  = &Activation_CPU::func_activation_ReLU;
			this->func_dactivation = &Activation_CPU::func_dactivation_ReLU;
			break;

			// Leakey-ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_LeakyReLU:
			this->func_activation  = &Activation_CPU::func_activation_LeakyReLU;
			this->func_dactivation = &Activation_CPU::func_dactivation_LeakyReLU;
			break;

			// SoftMax系
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
			this->func_activation  = &Activation_CPU::func_activation_SoftMax;
			this->func_dactivation = &Activation_CPU::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			this->func_activation  = &Activation_CPU::func_activation_SoftMax;
			this->func_dactivation = &Activation_CPU::func_dactivation_sigmoid_crossEntropy;
			break;
		}

		// 演算用のバッファを確保
		switch(this->layerData.layerStructure.ActivationType)
		{
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			this->lpCalculateSum.resize(this->GetInputDataStruct().z * this->GetInputDataStruct().y * this->GetInputDataStruct().x);
			break;
		default:
			this->lpCalculateSum.clear();
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Activation_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Activation_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->m_lppOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			memcpy(o_lppOutputBuffer, i_lppInputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());
			break;

		default:
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					// 活性化
					this->m_lppOutputBuffer[batchNum][inputNum] = (this->*func_activation)(this->m_lppInputBuffer[batchNum][inputNum]);
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
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					// 合計値を算出
					F32 sum = 0.0f;
					for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
					{
						sum += this->m_lppOutputBuffer[batchNum][outputNum];
					}

					// 値を合計値で割る
					for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
					{
						if(sum == 0.0f)
							this->m_lppOutputBuffer[batchNum][outputNum] = 1.0f / this->outputBufferCount;
						else
							this->m_lppOutputBuffer[batchNum][outputNum] /= sum;
					}
				}
			}
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			{
				U32 chSize = this->GetInputDataStruct().z * this->GetInputDataStruct().y * this->GetInputDataStruct().x;

				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					// 一時バッファクリア
					memset(&this->lpCalculateSum[0], 0, this->lpCalculateSum.size()*sizeof(F32));

					// 合計値を算出
					for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
					{
						for(U32 z=0; z<this->GetInputDataStruct().z; z++)
						{
							for(U32 y=0; y<this->GetInputDataStruct().y; y++)
							{
								for(U32 x=0; x<this->GetInputDataStruct().x; x++)
								{
									U32 offset = (((((ch*this->GetInputDataStruct().z+z)*this->GetInputDataStruct().y)+y)*this->GetInputDataStruct().x)+x);

									this->lpCalculateSum[offset] += this->m_lppOutputBuffer[batchNum][ch*chSize + offset];
								}
							}
						}
					}

					// 合計値で割る
					for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
					{
						for(U32 z=0; z<this->GetInputDataStruct().z; z++)
						{
							for(U32 y=0; y<this->GetInputDataStruct().y; y++)
							{
								for(U32 x=0; x<this->GetInputDataStruct().x; x++)
								{
									U32 offset = (((((ch*this->GetInputDataStruct().z+z)*this->GetInputDataStruct().y)+y)*this->GetInputDataStruct().x)+x);

									this->m_lppOutputBuffer[batchNum][ch*chSize + offset] /= this->lpCalculateSum[offset];
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


	//================================
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列の[GetOutputDataCount()]配列
		直前の計算結果を使用する */
	ErrorCode Activation_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 入力/出力バッファのアドレスを配列に格納
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
//				this->m_lppInputBuffer[batchNum]   = &i_lppInputBuffer[batchNum * this->inputBufferCount];
				this->m_lppDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
				this->m_lppOutputBuffer[batchNum]  = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
				this->m_lppDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
			}

			// 入力誤差を計算
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					this->m_lppDInputBuffer[batchNum][inputNum] = (this->*func_dactivation)(this->m_lppOutputBuffer[batchNum][inputNum]) * this->m_lppDOutputBuffer[batchNum][inputNum];
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Activation_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}



	//================================
	// 活性化関数
	//================================
	// lenear系
	F32 Activation_CPU::func_activation_lenear(F32 x)
	{
		return x;
	}
	F32 Activation_CPU::func_dactivation_lenear(F32 x)
	{
		return 1;
	}

	// sigmoid系
	F32 Activation_CPU::func_activation_sigmoid(F32 x)
	{
		return 1.0f / (1.0f + (F32)exp(-x));
	}
	F32 Activation_CPU::func_dactivation_sigmoid(F32 x)
	{
		return x * (1.0f - x);
	}

	F32 Activation_CPU::func_activation_sigmoid_crossEntropy(F32 x)
	{
		return 1.0f / (1.0f + (F32)exp(-x));
	}
	F32 Activation_CPU::func_dactivation_sigmoid_crossEntropy(F32 x)
	{
		return 1.0f;
	}

	// ReLU系
	F32 Activation_CPU::func_activation_ReLU(F32 x)
	{
		return x * (x > 0.0f);
	}
	F32 Activation_CPU::func_dactivation_ReLU(F32 x)
	{
		return 1.0f * (x > 0.0f);
	}

	// Leakey-ReLU系
	F32 Activation_CPU::func_activation_LeakyReLU(F32 x)
	{
		return x * ( (x>0.0f) + this->layerData.layerStructure.LeakyReLU_alpha * (x<=0.0f) );
	}
	F32 Activation_CPU::func_dactivation_LeakyReLU(F32 x)
	{
		return (x>0.0f) + this->layerData.layerStructure.LeakyReLU_alpha * (x<=0.0f);
	}

	// tanh系
	F32 Activation_CPU::func_activation_tanh(F32 x)
	{
		return tanh(x);
	}
	F32 Activation_CPU::func_dactivation_tanh(F32 x)
	{
		return 1.0f - x*x;
	}

	// SoftMax系
	F32 Activation_CPU::func_activation_SoftMax(F32 x)
	{
		return min(FLT_MAX, (F32)exp(x));	// 平均は別に行う
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
