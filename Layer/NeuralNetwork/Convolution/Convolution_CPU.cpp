//======================================
// 畳み込みニューラルネットワークの結合レイヤー
// CPU処理用
//======================================
#include"stdafx.h"

#include"Convolution_DATA.hpp"
#include"Convolution_FUNC.hpp"
#include"Convolution_Base.h"

#include"Convolution_CPU.h"
#include"Convolution_LayerData_CPU.h"

#include"Library/NeuralNetwork/Optimizer.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET_VECTOR(inX,inY,inZ,inCh,vector)	Gravisbell::CalculateOffset((vector).x,    (vector).y,    (vector).z, inCh, inX, inY, inZ)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Convolution_CPU::Convolution_CPU(Gravisbell::GUID guid, Convolution_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Convolution_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	neuronCount						(0)		/**< ニューロン数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
		,	temporaryMemoryManager			(i_temporaryMemoryManager)
	{
	}
	/** デストラクタ */
	Convolution_CPU::~Convolution_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Convolution_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Convolution_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	ILayerData& Convolution_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& Convolution_CPU::GetLayerData()const
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
	ErrorCode Convolution_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// ニューロン/バイアスの誤差を一時保存するバッファを作成
		this->lpDBias.resize(this->layerData.lpBias.size());
		this->lpDNeuron.resize(this->layerData.lpNeuron.size());

		// 入力誤差/出力誤差バッファを作成
		this->lppBatchDInputBuffer.resize(this->GetBatchSize());
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Convolution_CPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// ニューロン数を確認
		this->neuronCount = this->layerData.layerStructure.Output_Channel;
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// フィルタサイズ確認
		this->filterSize = this->layerData.layerStructure.FilterSize.x * this->layerData.layerStructure.FilterSize.y * this->layerData.layerStructure.FilterSize.z;
		if(this->filterSize == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// ニューロンバッファのサイズ確認
		if(this->layerData.lpNeuron.size() != this->neuronCount * this->filterSize * this->GetInputDataStruct().ch)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->layerData.lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファを作成
		this->lppBatchInputBuffer.resize(this->GetBatchSize());
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Convolution_CPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}



	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Convolution_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN && this->GetRuntimeParameterByStructure().UpdateWeigthWithOutputVariance)
		{
			U32 PROCTIME_MAX = 5;			// 実行最大値
			F32	VARIANCE_TOLERANCE = 0.1f;	// 分散交差(許容範囲)

			U32 procTime = 0;
			do
			{
				// 演算を実行
				ErrorCode err = this->Calculate_base(i_lppInputBuffer, o_lppOutputBuffer);
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
							average += this->lppBatchOutputBuffer[batchNum][outputNum];
						}
					}
					average /= (this->outputBufferCount * this->GetBatchSize());

					// 分散を求める
					for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					{
						for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
						{
							variance += (this->lppBatchOutputBuffer[batchNum][outputNum] - average) * (this->lppBatchOutputBuffer[batchNum][outputNum] - average);
						}
					}
					variance /= (this->outputBufferCount * this->GetBatchSize());
				}

				if( abs(variance - 1.0f) < VARIANCE_TOLERANCE)
					break;

				// 標準偏差で重みを割って更新する
				F32 deviation = sqrtf(variance);
				{
					for(U32 neuronNum=0; neuronNum<this->layerData.lpNeuron.size(); neuronNum++)
					{
						this->layerData.lpNeuron[neuronNum] /= deviation;
					}
					for(U32 neuronNum=0; neuronNum<this->layerData.lpBias.size(); neuronNum++)
					{
						this->layerData.lpBias[neuronNum] /= deviation;
					}
				}

				procTime++;
			}while(procTime < PROCTIME_MAX);
		}
		else
		{
			ErrorCode err = this->Calculate_base(i_lppInputBuffer, o_lppOutputBuffer);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	ErrorCode Convolution_CPU::Calculate_base(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
			this->lppBatchInputBuffer[batchNum] = &i_lppInputBuffer[batchNum * this->inputBufferCount];
		}

		// 畳みこみ結合処理
		for(unsigned int batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 neuronNum=0; neuronNum<(U32)this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(U32 convZ=0; convZ<this->GetOutputDataStruct().z; convZ++)
				{
					for(U32 convY=0; convY<this->GetOutputDataStruct().y; convY++)
					{
						for(U32 convX=0; convX<this->GetOutputDataStruct().x; convX++)
						{
							U32 outputOffset = this->GetOutputDataStruct().POSITION_TO_OFFSET(convX,convY,convZ,neuronNum);

							// 一時保存用のバッファを作成
							F32 tmp = 0.0f;

							// フィルタを処理する
							for(U32 chNum=0; chNum<this->GetInputDataStruct().ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)(convZ * this->layerData.layerStructure.Stride.z + filterZ*this->layerData.layerStructure.Dilation.z - this->layerData.layerStructure.Padding.z);
									if((U32)inputZ >= this->GetInputDataStruct().z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)(convY * this->layerData.layerStructure.Stride.y + filterY*this->layerData.layerStructure.Dilation.y - this->layerData.layerStructure.Padding.y);
										if((U32)inputY >= this->GetInputDataStruct().y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)(convX * this->layerData.layerStructure.Stride.x + filterX*this->layerData.layerStructure.Dilation.x - this->layerData.layerStructure.Padding.x);
											if((U32)inputX >= this->GetInputDataStruct().x)
												continue;

											const S32 inputOffset  = this->GetInputDataStruct().POSITION_TO_OFFSET(inputX,inputY,inputZ, chNum);
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize);

											tmp += this->layerData.lppNeuron[neuronNum][filterOffset] * this->lppBatchInputBuffer[batchNum][inputOffset];
										}
									}
								}
							}
							// バイアスを追加
							tmp += this->layerData.lpBias[neuronNum];

							// 計算結果を格納する
							this->lppBatchOutputBuffer[batchNum][outputOffset] = tmp;

						}
					}
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
	ErrorCode Convolution_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
			this->lppBatchInputBuffer[batchNum] = &i_lppInputBuffer[batchNum * this->inputBufferCount];
		}

		// 出力誤差バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
		
		// 入力誤差バッファのアドレスを配列に格納
		if(o_lppDInputBuffer)
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->lppBatchDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
			
			// 入力誤差バッファを初期化
			memset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());
		}

		
		// ニューロンとバイアスの変化量を初期化
		memset(&this->lpDBias[0], 0, sizeof(F32)*this->lpDBias.size());
		memset(&this->lpDNeuron[0], 0, sizeof(F32)*this->lpDNeuron.size());

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			// 入力誤差計算
			for(S32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(U32 convZ=0; convZ<this->GetOutputDataStruct().z; convZ++)
				{
					for(U32 convY=0; convY<this->GetOutputDataStruct().y; convY++)
					{
						for(U32 convX=0; convX<this->GetOutputDataStruct().x; convX++)
						{
							U32 outputOffet = this->GetOutputDataStruct().POSITION_TO_OFFSET(convX,convY,convZ,neuronNum);
							F32 dOutput = this->lppBatchDOutputBuffer[batchNum][outputOffet];

							// フィルタを処理する
							for(U32 chNum=0; chNum<this->GetInputDataStruct().ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)((convZ * this->layerData.layerStructure.Stride.z) - this->layerData.layerStructure.Padding.z + filterZ*this->layerData.layerStructure.Dilation.z);
									if((U32)inputZ>=this->GetInputDataStruct().z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)((convY * this->layerData.layerStructure.Stride.y) - this->layerData.layerStructure.Padding.y + filterY*this->layerData.layerStructure.Dilation.y);
										if((U32)inputY>=this->GetInputDataStruct().y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)((convX * this->layerData.layerStructure.Stride.x) - this->layerData.layerStructure.Padding.x + filterX*this->layerData.layerStructure.Dilation.x);
											if((U32)inputX>=this->GetInputDataStruct().x)
												continue;

											const S32 inputOffset  = this->GetInputDataStruct().POSITION_TO_OFFSET(inputX,inputY,inputZ, chNum);
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize);

											if(o_lppDInputBuffer)
												this->lppBatchDInputBuffer[batchNum][inputOffset] += this->layerData.lppNeuron[neuronNum][filterOffset] * dOutput;

											// ニューロンの重み変化量を追加
											this->lpDNeuron[neuronNum*this->filterSize*this->GetInputDataStruct().ch + filterOffset] += this->lppBatchInputBuffer[batchNum][inputOffset] * dOutput;
										}
									}
								}
							}

							// バイアスの重み変化量を追加
							this->lpDBias[neuronNum] += dOutput;

						}
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
	ErrorCode Convolution_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// 学習差分の反映
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(&this->layerData.lpBias[0], &this->lpDBias[0]);
		if(this->layerData.m_pOptimizer_neuron)
			this->layerData.m_pOptimizer_neuron->UpdateParameter(&this->layerData.lpNeuron[0], &this->lpDNeuron[0]);


		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
