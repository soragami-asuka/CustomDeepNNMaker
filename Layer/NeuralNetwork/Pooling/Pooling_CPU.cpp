//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"Pooling_DATA.hpp"
#include"Pooling_FUNC.hpp"
#include"Pooling_Base.h"

#include"Pooling_CPU.h"
#include"Pooling_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Pooling_CPU::Pooling_CPU(Gravisbell::GUID guid, Pooling_LayerData_CPU& i_layerData)
		:	Pooling_Base	(guid)
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
		,	m_lppInputBuffer				(NULL)	/**< 演算時の入力データ */
		,	m_lppDOutputBufferPrev			(NULL)	/**< 入力誤差計算時の出力誤差データ */
	{
	}
	/** デストラクタ */
	Pooling_CPU::~Pooling_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Pooling_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Pooling_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Pooling_LayerData_Base& Pooling_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const Pooling_LayerData_Base& Pooling_CPU::GetLayerData()const
	{
		return this->layerData;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 Pooling_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
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
	ErrorCode Pooling_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

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
	ErrorCode Pooling_CPU::PreProcessCalculate(unsigned int batchSize)
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


		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->batchSize);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpOutputBuffer[batchNum].resize(this->outputBufferCount);
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum][0];
		}


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Pooling_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Pooling_CPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Pooling_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		this->m_lppInputBuffer = i_lppInputBuffer;

		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 ch=0; ch<this->layerData.outputDataStruct.ch; ch++)
			{
				for(U32 outputZ=0; outputZ<this->layerData.outputDataStruct.z; outputZ++)
				{
					for(U32 outputY=0; outputY<this->layerData.outputDataStruct.y; outputY++)
					{
						for(U32 outputX=0; outputX<this->layerData.outputDataStruct.x; outputX++)
						{
							// 最大値を調べる
							F32 maxValue = -FLT_MAX;
							for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
							{
								for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
								{
									for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
									{
										U32 inputX = outputX * this->layerData.layerStructure.FilterSize.x + filterX;
										U32 inputY = outputY * this->layerData.layerStructure.FilterSize.y + filterY;
										U32 inputZ = outputZ * this->layerData.layerStructure.FilterSize.z + filterZ;
										if(inputX >= this->layerData.inputDataStruct.x)
											continue;
										if(inputY >= this->layerData.inputDataStruct.y)
											continue;
										if(inputZ >= this->layerData.inputDataStruct.z)
											continue;

										U32 inputOffset = POSITION_TO_OFFSET_STRUCT(
																inputX,
																inputY,
																inputZ,
																ch,
																this->layerData.inputDataStruct);

										maxValue = max(maxValue, this->m_lppInputBuffer[batchNum][inputOffset]);
									}
								}
							}
							
							U32 outputOffset = POSITION_TO_OFFSET_STRUCT(outputX,outputY,outputZ,ch, this->layerData.outputDataStruct);
							this->lpOutputBuffer[batchNum][outputOffset] = maxValue;
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
	CONST_BATCH_BUFFER_POINTER Pooling_CPU::GetOutputBuffer()const
	{
		return &this->lppBatchOutputBuffer[0];
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode Pooling_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseOutputBuffer = this->GetOutputBuffer();

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpOutputBuffer[batchNum], lppUseOutputBuffer[batchNum], sizeof(F32)*outputBufferCount);
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
	ErrorCode Pooling_CPU::CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBufferPrev)
	{
		this->m_lppDOutputBufferPrev = i_lppDOutputBufferPrev;
		
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			// バッファをクリア
			memset(&this->lpDInputBuffer[batchNum][0], 0, this->inputBufferCount * sizeof(F32));
			
			for(U32 ch=0; ch<this->layerData.outputDataStruct.ch; ch++)
			{
				for(U32 outputZ=0; outputZ<this->layerData.outputDataStruct.z; outputZ++)
				{
					for(U32 outputY=0; outputY<this->layerData.outputDataStruct.y; outputY++)
					{
						for(U32 outputX=0; outputX<this->layerData.outputDataStruct.x; outputX++)
						{
							U32 outputOffset = POSITION_TO_OFFSET_STRUCT(outputX,outputY,outputZ,ch, this->layerData.outputDataStruct);

							// 最大値を調べる
							for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
							{
								for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
								{
									for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
									{
										U32 inputX = outputX * this->layerData.layerStructure.FilterSize.x + filterX;
										U32 inputY = outputY * this->layerData.layerStructure.FilterSize.y + filterY;
										U32 inputZ = outputZ * this->layerData.layerStructure.FilterSize.z + filterZ;
										if(inputX >= this->layerData.inputDataStruct.x)
											continue;
										if(inputY >= this->layerData.inputDataStruct.y)
											continue;
										if(inputZ >= this->layerData.inputDataStruct.z)
											continue;

										U32 inputOffset = POSITION_TO_OFFSET_STRUCT(
																inputX,
																inputY,
																inputZ,
																ch,
																this->layerData.inputDataStruct);

										if(this->m_lppInputBuffer[batchNum][inputOffset] == this->lpOutputBuffer[batchNum][outputOffset])
											this->lpDInputBuffer[batchNum][inputOffset] = this->m_lppDOutputBufferPrev[batchNum][outputOffset];
									}
								}
							}

						}
					}
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習差分をレイヤーに反映させる.
		入力信号、出力信号は直前のCalculateの値を参照する.
		出力誤差差分、入力誤差差分は直前のCalculateLearnErrorの値を参照する. */
	ErrorCode Pooling_CPU::ReflectionLearnError(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Pooling_CPU::GetDInputBuffer()const
	{
		return &this->lppBatchDInputBuffer[0];
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode Pooling_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseDInputBuffer = this->GetDInputBuffer();

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpDInputBuffer[batchNum], lppUseDInputBuffer[batchNum], sizeof(F32)*inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
