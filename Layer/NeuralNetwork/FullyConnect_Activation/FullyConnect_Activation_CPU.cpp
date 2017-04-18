//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"FullyConnect_Activation_DATA.hpp"
#include"FullyConnect_Activation_FUNC.hpp"
#include"FullyConnect_Activation_Base.h"

#include"FullyConnect_Activation_CPU.h"
#include"FullyConnect_Activation_LayerData_CPU.h"

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
	FullyConnect_Activation_CPU::FullyConnect_Activation_CPU(Gravisbell::GUID guid, FullyConnect_Activation_LayerData_CPU& i_layerData)
		:	FullyConnect_Activation_Base	(guid)
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	neuronCount						(0)		/**< ニューロン数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
		,	m_lppInputBuffer				(NULL)	/**< 演算時の入力データ */
		,	m_lppDOutputBufferPrev			(NULL)	/**< 入力誤差計算時の出力誤差データ */
		,	onUseDropOut					(false)
		,	func_activation					(func_activation_sigmoid)
		,	func_dactivation				(func_dactivation_sigmoid)
	{
	}
	/** デストラクタ */
	FullyConnect_Activation_CPU::~FullyConnect_Activation_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 FullyConnect_Activation_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode FullyConnect_Activation_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	FullyConnect_Activation_LayerData_Base& FullyConnect_Activation_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const FullyConnect_Activation_LayerData_Base& FullyConnect_Activation_CPU::GetLayerData()const
	{
		return this->layerData;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 FullyConnect_Activation_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
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
	ErrorCode FullyConnect_Activation_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;
		
		// 出力誤差バッファを作成
		this->lpDOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpDOutputBuffer[batchNum].resize(this->outputBufferCount);
		}

		// 入力差分バッファを作成
		this->lpDInputBuffer.resize(this->batchSize);
		this->lppBatchDInputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpDInputBuffer[batchNum].resize(this->inputBufferCount);
			this->lppBatchDInputBuffer[batchNum] = &this->lpDInputBuffer[batchNum][0];
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FullyConnect_Activation_CPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

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

		// ニューロンバッファのサイズ確認
		if(this->layerData.lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->layerData.lppNeuron[0].size() != this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;


		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->batchSize);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpOutputBuffer[batchNum].resize(this->outputBufferCount);
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum][0];
		}

		// ドロップアウト処理を未使用に変更
		this->onUseDropOut = false;
		this->lppDropOutBuffer.clear();

		// 活性化関数を設定
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_lenear:
			this->func_activation  = ::func_activation_lenear;
			this->func_dactivation = ::func_dactivation_lenear;
			break;

			// Sigmoid
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_sigmoid:
		default:
			this->func_activation  = ::func_activation_sigmoid;
			this->func_dactivation = ::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_sigmoid_crossEntropy:
			this->func_activation  = ::func_activation_sigmoid_crossEntropy;
			this->func_dactivation = ::func_dactivation_sigmoid_crossEntropy;
			break;

			// ReLU
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU:
			this->func_activation  = ::func_activation_ReLU;
			this->func_dactivation = ::func_dactivation_ReLU;
			break;

			// Softmax
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_softmax:
			this->func_activation  = ::func_activation_SoftMax;
			this->func_dactivation = ::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_softmax_crossEntropy:
			this->func_activation  = ::func_activation_SoftMax;
			this->func_dactivation = ::func_dactivation_sigmoid_crossEntropy;
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FullyConnect_Activation_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		// ドロップアウト
		{
			S32 dropOutRate = (S32)(this->layerData.layerStructure.DropOut * RAND_MAX);

			if(dropOutRate > 0)
			{
				this->onUseDropOut = true;
				if(this->lppDropOutBuffer.empty())
				{
					// バッファの確保
					this->lppDropOutBuffer.resize(this->neuronCount);
					for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
						this->lppDropOutBuffer[neuronNum].resize(this->inputBufferCount);
				}

				// バッファに1or0を入力
				// 1 : DropOutしない
				// 0 : DropOutする
				for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
					{
						if(rand() < dropOutRate)	// ドロップアウトする
							this->lppDropOutBuffer[neuronNum][inputNum] = 0.0f;
						else
							this->lppDropOutBuffer[neuronNum][inputNum] = 1.0f;
					}
				}
			}
			else
			{
				this->onUseDropOut = false;
				this->lppDropOutBuffer.clear();
			}
		}
		// 学習係数
		{
			auto pItem = dynamic_cast<const Gravisbell::SettingData::Standard::IItem_Float*>(data.GetItemByID(L"LearnCoeff"));
			if(pItem)
				this->learnData.LearnCoeff = pItem->GetValue();
			else
				this->learnData.LearnCoeff = 1.0f;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FullyConnect_Activation_CPU::PreProcessCalculateLoop()
	{
		this->onUseDropOut = false;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FullyConnect_Activation_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		this->m_lppInputBuffer = i_lppInputBuffer;

		for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				float tmp = 0;

				// ニューロンの値を加算
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					if(this->onUseDropOut)
						tmp += i_lppInputBuffer[batchNum][inputNum] * this->layerData.lppNeuron[neuronNum][inputNum] * this->lppDropOutBuffer[neuronNum][inputNum];
					else
						tmp += i_lppInputBuffer[batchNum][inputNum] * this->layerData.lppNeuron[neuronNum][inputNum];
				}
				tmp += this->layerData.lpBias[neuronNum];

				if(!this->onUseDropOut && this->layerData.layerStructure.DropOut > 0.0f)
					tmp *= (1.0f - this->layerData.layerStructure.DropOut);

				// 活性化
				this->lpOutputBuffer[batchNum][neuronNum] = this->func_activation(tmp);

#ifdef _DEBUG
				if(isnan(this->lpOutputBuffer[batchNum][neuronNum]))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
			}
		}

		// 活性化後のデータを合計値で割る
		switch(this->layerData.layerStructure.ActivationType)
		{
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_softmax:
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_softmax_crossEntropy:
			{
				for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					// 平均を算出
					F32 sum = 0.0f;
					for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						sum += this->lpOutputBuffer[batchNum][neuronNum];
					}

					// 値を平均値で割る
					for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						if(sum == 0.0f)
							this->lpOutputBuffer[batchNum][neuronNum] = 1.0f / neuronNum;
						else
							this->lpOutputBuffer[batchNum][neuronNum] /= sum;
					
#ifdef _DEBUG
						if(isnan(this->lpOutputBuffer[batchNum][neuronNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
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
	CONST_BATCH_BUFFER_POINTER FullyConnect_Activation_CPU::GetOutputBuffer()const
	{
		return &this->lppBatchOutputBuffer[0];
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode FullyConnect_Activation_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpOutputBuffer[batchNum], this->lppBatchOutputBuffer[batchNum], sizeof(F32)*outputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// 学習処理
	//================================
	/** 学習誤差を計算する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FullyConnect_Activation_CPU::CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBufferPrev)
	{
		this->m_lppDOutputBufferPrev = i_lppDOutputBufferPrev;

		// 出力誤差を計算
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				this->lpDOutputBuffer[batchNum][neuronNum] = this->func_dactivation(this->lpOutputBuffer[batchNum][neuronNum]) * i_lppDOutputBufferPrev[batchNum][neuronNum];

#ifdef _DEBUG
				if(isnan(this->lpDOutputBuffer[batchNum][neuronNum]))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
			}
		}


#if 0
		for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// バイアス更新
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					 sumDOutput += this->lpDOutputBuffer[batchNum][neuronNum];
				}

#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

				this->layerData.lpBias[neuronNum] += this->learnData.LearnCoeff * sumDOutput;
			}

			// 入力対応ニューロン更新
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				if(this->onUseDropOut && this->lppDropOutBuffer[neuronNum][inputNum] == 0.0f)
					continue;

				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					sumDOutput += this->m_lppInputBuffer[batchNum][inputNum] * this->lpDOutputBuffer[batchNum][neuronNum];
				}

#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

				this->layerData.lppNeuron[neuronNum][inputNum] += this->learnData.LearnCoeff * sumDOutput;
			}
		}
#endif



		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			// 入力誤差差分を計算
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				float tmp = 0.0f;

				for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					if(this->onUseDropOut)
						tmp += this->lpDOutputBuffer[batchNum][neuronNum] * this->layerData.lppNeuron[neuronNum][inputNum] * this->lppDropOutBuffer[neuronNum][inputNum];
					else
						tmp += this->lpDOutputBuffer[batchNum][neuronNum] * this->layerData.lppNeuron[neuronNum][inputNum];
				}

				this->lpDInputBuffer[batchNum][inputNum] = tmp;

#ifdef _DEBUG
				if(isnan(this->lpDInputBuffer[batchNum][inputNum]))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習差分をレイヤーに反映させる.
		入力信号、出力信号は直前のCalculateの値を参照する.
		出力誤差差分、入力誤差差分は直前のCalculateLearnErrorの値を参照する. */
	ErrorCode FullyConnect_Activation_CPU::ReflectionLearnError(void)
	{
#if 1
		for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// バイアス更新
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					 sumDOutput += this->lpDOutputBuffer[batchNum][neuronNum];
				}

#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

				this->layerData.lpBias[neuronNum] += this->learnData.LearnCoeff * sumDOutput;
			}

			// 入力対応ニューロン更新
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				if(this->onUseDropOut && this->lppDropOutBuffer[neuronNum][inputNum] == 0.0f)
					continue;

				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					sumDOutput += this->m_lppInputBuffer[batchNum][inputNum] * this->lpDOutputBuffer[batchNum][neuronNum];
				}

#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

				this->layerData.lppNeuron[neuronNum][inputNum] += this->learnData.LearnCoeff * sumDOutput;
			}
		}
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER FullyConnect_Activation_CPU::GetDInputBuffer()const
	{
		return &this->lppBatchDInputBuffer[0];
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode FullyConnect_Activation_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpDInputBuffer[batchNum], this->lppBatchDInputBuffer[batchNum], sizeof(F32)*inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
