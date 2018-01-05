//======================================
// 畳み込みニューラルネットワークの結合レイヤー
// CPU処理用
//======================================
#include"stdafx.h"

#include"UpSampling_DATA.hpp"
#include"UpSampling_FUNC.hpp"
#include"UpSampling_Base.h"

#include"UpSampling_CPU.h"
#include"UpSampling_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)
#define POSITION_TO_OFFSET_VECTOR(inX,inY,inZ,inCh,vector,chSize)	POSITION_TO_OFFSET(inX, inY, inZ, inCh, vector.x,    vector.y,    vector.z,    chSize)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	UpSampling_CPU::UpSampling_CPU(Gravisbell::GUID guid, UpSampling_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	UpSampling_Base					(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
	{
	}
	/** デストラクタ */
	UpSampling_CPU::~UpSampling_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 UpSampling_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode UpSampling_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	UpSampling_LayerData_Base& UpSampling_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const UpSampling_LayerData_Base& UpSampling_CPU::GetLayerData()const
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
	ErrorCode UpSampling_CPU::PreProcessLearn()
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
	ErrorCode UpSampling_CPU::PreProcessCalculate()
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
	ErrorCode UpSampling_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode UpSampling_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		// 出力バッファをクリア
		memset(o_lppOutputBuffer, 0, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
			{
				for(U32 inputZ=0; inputZ<this->GetInputDataStruct().z; inputZ++)
				{
					for(U32 inputY=0; inputY<this->GetInputDataStruct().y; inputY++)
					{
						for(U32 inputX=0; inputX<this->GetInputDataStruct().x; inputX++)
						{
							U32 inputOffset = POSITION_TO_OFFSET_STRUCT(inputX, inputY, inputZ, ch, this->GetInputDataStruct());

							switch(this->layerData.layerStructure.PaddingType)
							{
							case UpSampling::LayerStructure::PaddingType_value:
								{
									for(S32 offsetZ=0; offsetZ<this->layerData.layerStructure.UpScale.z; offsetZ++)
									{
										for(S32 offsetY=0; offsetY<this->layerData.layerStructure.UpScale.y; offsetY++)
										{
											for(S32 offsetX=0; offsetX<this->layerData.layerStructure.UpScale.x; offsetX++)
											{
												U32 outputOffset = POSITION_TO_OFFSET_STRUCT(
													inputX*this->layerData.layerStructure.UpScale.x + offsetX,
													inputY*this->layerData.layerStructure.UpScale.y + offsetY,
													inputZ*this->layerData.layerStructure.UpScale.z + offsetZ,
													ch,
													this->GetOutputDataStruct());

												this->lppBatchOutputBuffer[batchNum][outputOffset] = this->lppBatchInputBuffer[batchNum][inputOffset];
											}
										}
									}
								}
								break;
							case UpSampling::LayerStructure::PaddingType_zero:
								{
									U32 outputOffset = POSITION_TO_OFFSET_STRUCT(
										inputX*this->layerData.layerStructure.UpScale.x + 0,
										inputY*this->layerData.layerStructure.UpScale.y + 0,
										inputZ*this->layerData.layerStructure.UpScale.z + 0,
										ch,
										this->GetOutputDataStruct());

									this->lppBatchOutputBuffer[batchNum][outputOffset] = this->lppBatchInputBuffer[batchNum][inputOffset];
								}
								break;
							}
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
	ErrorCode UpSampling_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 入力誤差/出力誤差バッファのアドレスを配列に格納
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
				this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
			}

			// 入力誤差バッファを初期化
			memset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

		
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
				{
					for(U32 inputZ=0; inputZ<this->GetInputDataStruct().z; inputZ++)
					{
						for(U32 inputY=0; inputY<this->GetInputDataStruct().y; inputY++)
						{
							for(U32 inputX=0; inputX<this->GetInputDataStruct().x; inputX++)
							{
								U32 inputOffset = POSITION_TO_OFFSET_STRUCT(inputX, inputY, inputZ, ch, this->GetInputDataStruct());

								switch(this->layerData.layerStructure.PaddingType)
								{
								case UpSampling::LayerStructure::PaddingType_value:
									{
										for(S32 offsetZ=0; offsetZ<this->layerData.layerStructure.UpScale.z; offsetZ++)
										{
											for(S32 offsetY=0; offsetY<this->layerData.layerStructure.UpScale.y; offsetY++)
											{
												for(S32 offsetX=0; offsetX<this->layerData.layerStructure.UpScale.x; offsetX++)
												{
													U32 outputOffset = POSITION_TO_OFFSET_STRUCT(
														inputX*this->layerData.layerStructure.UpScale.x + offsetX,
														inputY*this->layerData.layerStructure.UpScale.y + offsetY,
														inputZ*this->layerData.layerStructure.UpScale.z + offsetZ,
														ch,
														this->GetOutputDataStruct());


													this->lppBatchDInputBuffer[batchNum][inputOffset] += this->lppBatchDOutputBuffer[batchNum][outputOffset];
												}
											}
										}
									}
									break;
								case UpSampling::LayerStructure::PaddingType_zero:
									{
										U32 outputOffset = POSITION_TO_OFFSET_STRUCT(
											inputX*this->layerData.layerStructure.UpScale.x + 0,
											inputY*this->layerData.layerStructure.UpScale.y + 0,
											inputZ*this->layerData.layerStructure.UpScale.z + 0,
											ch,
											this->GetOutputDataStruct());

										this->lppBatchDInputBuffer[batchNum][inputOffset] = this->lppBatchDOutputBuffer[batchNum][outputOffset];
									}
									break;
								}

							}
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
	ErrorCode UpSampling_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}



} // Gravisbell;
} // Layer;
} // NeuralNetwork;
