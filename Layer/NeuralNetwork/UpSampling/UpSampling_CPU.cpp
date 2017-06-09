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
	UpSampling_CPU::UpSampling_CPU(Gravisbell::GUID guid, UpSampling_LayerData_CPU& i_layerData)
		:	UpSampling_Base	(guid)
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
	ErrorCode UpSampling_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 出力誤差バッファ受け取り用のアドレス配列を作成する
		this->m_lppDOutputBuffer.resize(batchSize);
		// 入力誤差バッファ受け取り用のアドレス配列を作成する
		this->m_lppDInputBuffer.resize(batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode UpSampling_CPU::PreProcessCalculate(unsigned int batchSize)
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
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum * this->outputBufferCount];
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode UpSampling_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode UpSampling_CPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode UpSampling_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];

		// 出力バッファをクリア
		memset(&this->lpOutputBuffer[0], 0, sizeof(F32)*this->lpOutputBuffer.size());

		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
			{
				for(U32 inputZ=0; inputZ<this->layerData.inputDataStruct.z; inputZ++)
				{
					for(U32 inputY=0; inputY<this->layerData.inputDataStruct.y; inputY++)
					{
						for(U32 inputX=0; inputX<this->layerData.inputDataStruct.x; inputX++)
						{
							U32 inputOffset = POSITION_TO_OFFSET_STRUCT(inputX, inputY, inputZ, ch, this->layerData.inputDataStruct);

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
													this->layerData.outputDataStruct);

												this->lppBatchOutputBuffer[batchNum][outputOffset] = this->m_lppInputBuffer[batchNum][inputOffset];
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
										this->layerData.outputDataStruct);

									this->lppBatchOutputBuffer[batchNum][outputOffset] = this->m_lppInputBuffer[batchNum][inputOffset];
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


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER UpSampling_CPU::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode UpSampling_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	ErrorCode UpSampling_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		// 出力誤差バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppDOutputBuffer[batchNum] = &i_lpDOutputBufferPrev[batchNum * this->outputBufferCount];

		// 入力誤差計算
		this->m_lpDInputBuffer = o_lppDInputBuffer;
		if(o_lppDInputBuffer)
		{
			// 入力誤差バッファのアドレスを配列に格納
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				this->m_lppDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];

			// 入力誤差バッファを初期化
			memset(this->m_lpDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->batchSize);

		
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
				{
					for(U32 inputZ=0; inputZ<this->layerData.inputDataStruct.z; inputZ++)
					{
						for(U32 inputY=0; inputY<this->layerData.inputDataStruct.y; inputY++)
						{
							for(U32 inputX=0; inputX<this->layerData.inputDataStruct.x; inputX++)
							{
								U32 inputOffset = POSITION_TO_OFFSET_STRUCT(inputX, inputY, inputZ, ch, this->layerData.inputDataStruct);

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
														this->layerData.outputDataStruct);


													this->m_lppDInputBuffer[batchNum][inputOffset] += this->m_lppDOutputBuffer[batchNum][outputOffset];
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
											this->layerData.outputDataStruct);

										this->m_lppDInputBuffer[batchNum][inputOffset] = this->m_lppDOutputBuffer[batchNum][outputOffset];
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


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER UpSampling_CPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer;
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode UpSampling_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		memcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*inputBufferCount*this->batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
