//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"Activation_Discriminator_DATA.hpp"
#include"Activation_Discriminator_FUNC.hpp"
#include"Activation_Discriminator_Base.h"

#include"Activation_Discriminator_CPU.h"
#include"Activation_Discriminator_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace
{
	//================================
	// 活性化関数
	//================================
	// SoftMax系
	F32 func_Activation_Discriminator_SoftMax(F32 x)
	{
		return (F32)exp(x);	// 平均は別に行う
	}
	F32 func_dactivation_Discriminator_SoftMax_crossEntropy(F32 x)
	{
		return 1.0f;
	}
}


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Activation_Discriminator_CPU::Activation_Discriminator_CPU(Gravisbell::GUID guid, Activation_Discriminator_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Activation_Discriminator_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
		,	func_Activation					(func_Activation_Discriminator_SoftMax)
		,	func_dActivation				(func_dactivation_Discriminator_SoftMax_crossEntropy)
	{
	}
	/** デストラクタ */
	Activation_Discriminator_CPU::~Activation_Discriminator_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Activation_Discriminator_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Activation_Discriminator_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	ILayerData& Activation_Discriminator_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& Activation_Discriminator_CPU::GetLayerData()const
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
	ErrorCode Activation_Discriminator_CPU::PreProcessLearn()
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
	ErrorCode Activation_Discriminator_CPU::PreProcessCalculate()
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
	ErrorCode Activation_Discriminator_CPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Activation_Discriminator_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		std::vector<F32> lpOutputBuffer(this->inputBufferCount);
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			// 活性化して合計値を算出
			F32 sum = 0.0f;
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				// 活性化
				lpOutputBuffer[inputNum] = this->func_Activation(this->lppBatchInputBuffer[batchNum][inputNum]);
				sum += lpOutputBuffer[inputNum];
			}

			// 値を合計値で割る
			this->lppBatchOutputBuffer[batchNum][0] = lpOutputBuffer[0] / sum;
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
	ErrorCode Activation_Discriminator_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 入力誤差/出力誤差バッファのアドレスを配列に格納
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
				this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
				this->lppBatchOutputBuffer[batchNum]  = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
			}

			// 入力誤差を計算
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
			//	this->lppBatchDInputBuffer[batchNum][0] = this->func_dActivation(       this->lppBatchOutputBuffer[batchNum][0]) *  this->lppBatchDOutputBuffer[batchNum][0];
			//	this->lppBatchDInputBuffer[batchNum][1] = this->func_dActivation(1.0f - this->lppBatchOutputBuffer[batchNum][0]) * -this->lppBatchDOutputBuffer[batchNum][0];
				this->lppBatchDInputBuffer[batchNum][0] =  this->lppBatchDOutputBuffer[batchNum][0];
				this->lppBatchDInputBuffer[batchNum][1] = -this->lppBatchDOutputBuffer[batchNum][0];
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Activation_Discriminator_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
