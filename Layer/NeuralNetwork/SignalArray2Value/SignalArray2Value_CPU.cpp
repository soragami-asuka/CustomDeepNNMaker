//======================================
// 出力信号分割レイヤー
// CPU処理用
//======================================
#include"stdafx.h"

#include<algorithm>

#include"SignalArray2Value_DATA.hpp"
#include"SignalArray2Value_FUNC.hpp"
#include"SignalArray2Value_Base.h"

#include"SignalArray2Value_CPU.h"
#include"SignalArray2Value_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	SignalArray2Value_CPU::SignalArray2Value_CPU(Gravisbell::GUID guid, SignalArray2Value_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	SignalArray2Value_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
	{
	}
	/** デストラクタ */
	SignalArray2Value_CPU::~SignalArray2Value_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 SignalArray2Value_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode SignalArray2Value_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	SignalArray2Value_LayerData_Base& SignalArray2Value_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const SignalArray2Value_LayerData_Base& SignalArray2Value_CPU::GetLayerData()const
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
	ErrorCode SignalArray2Value_CPU::PreProcessLearn()
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
	ErrorCode SignalArray2Value_CPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// チャンネルのバッファサイズを保存
		this->channelSize = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;

		// 入力/出力バッファ保存用のアドレス配列を作成
		this->lppBatchInputBuffer.resize(this->GetBatchSize(), NULL);
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode SignalArray2Value_CPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode SignalArray2Value_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		// 入力バッファの指定チャンネルを出力バッファにコピー
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 z=0; z<this->GetInputDataStruct().z; z++)
			{
				for(U32 y=0; y<this->GetInputDataStruct().y; y++)
				{
					for(U32 x=0; x<this->GetInputDataStruct().x; x++)
					{
						U32 inputOffset = this->GetInputDataStruct().POSITION_TO_OFFSET(x, y, z, 0);
						F32 maxValue = this->lppBatchInputBuffer[batchNum][inputOffset];
						U32 maxCh = 0;

						for(U32 ch=1; ch<this->GetInputDataStruct().ch; ch++)
						{
							inputOffset  = this->GetInputDataStruct().POSITION_TO_OFFSET(x, y, z, ch);
							F32 value = this->lppBatchInputBuffer[batchNum][inputOffset];
							if(value > maxValue)
							{
								maxValue = value;
								maxCh = ch;
							}
						}

						U32 outputOffset = this->GetOutputDataStruct().POSITION_TO_OFFSET(x, y, z, 0);
						this->lppBatchOutputBuffer[batchNum][outputOffset] =
							((float)maxCh / this->GetInputDataStruct().ch)
							* (this->layerData.layerStructure.outputMaxValue - this->layerData.layerStructure.outputMinValue)
							+ this->layerData.layerStructure.outputMinValue;
					}
				}
			}

		}

#if _DEBUG
		std::vector<F32> lpInputBuffer(this->inputBufferCount * this->GetBatchSize());
		memcpy(&lpInputBuffer[0], i_lppInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize());

		std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
		memcpy(&lpOutputBuffer[0], o_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize());
#endif

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
	ErrorCode SignalArray2Value_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 入出力誤差バッファのアドレスを配列に格納
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
				this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
				this->lppBatchOutputBuffer[batchNum]  = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
			}

			// 入力誤差バッファをクリア
			memset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 z=0; z<this->GetInputDataStruct().z; z++)
				{
					for(U32 y=0; y<this->GetInputDataStruct().y; y++)
					{
						for(U32 x=0; x<this->GetInputDataStruct().x; x++)
						{
							U32 outputOffset = this->GetInputDataStruct().POSITION_TO_OFFSET(x, y, z, 0);

							// 正解を求める
							F32 trueValue = this->lppBatchOutputBuffer[batchNum][outputOffset] + this->lppBatchDOutputBuffer[batchNum][outputOffset];
							U32 truePos = std::max<U32>(0, std::min<U32>(this->GetInputDataStruct().ch-1, (U32)((trueValue - this->layerData.layerStructure.outputMinValue) / (this->layerData.layerStructure.outputMaxValue - this->layerData.layerStructure.outputMinValue) * this->GetInputDataStruct().ch + 0.5f)));

							for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
							{
								U32 inputOffset  = this->GetInputDataStruct().POSITION_TO_OFFSET(x, y, z, ch);

								this->lppBatchDInputBuffer[batchNum][inputOffset] = (ch == truePos) - this->lppBatchInputBuffer[batchNum][inputOffset];
							}
						}
					}
				}
#if _DEBUG
				std::vector<F32> lpDOutputBuffer(this->outputBufferCount);
				memcpy(&lpDOutputBuffer[0], this->lppBatchDOutputBuffer[batchNum], sizeof(F32) * this->outputBufferCount);

				std::vector<F32> lpDInputBuffer(this->inputBufferCount);
				memcpy(&lpDInputBuffer[0], this->lppBatchDInputBuffer[batchNum], sizeof(F32) * this->inputBufferCount);
#endif

			}

		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode SignalArray2Value_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
