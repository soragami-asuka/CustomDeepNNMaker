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

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)
#define POSITION_TO_OFFSET_VECTOR(inX,inY,inZ,inCh,vector,chSize)	POSITION_TO_OFFSET(inX, inY, inZ, inCh, vector.x,    vector.y,    vector.z,    chSize)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Convolution_CPU::Convolution_CPU(Gravisbell::GUID guid, Convolution_LayerData_CPU& i_layerData)
		:	Convolution_Base	(guid)
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	neuronCount						(0)		/**< ニューロン数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
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
	Convolution_LayerData_Base& Convolution_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const Convolution_LayerData_Base& Convolution_CPU::GetLayerData()const
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
	ErrorCode Convolution_CPU::PreProcessLearn(unsigned int batchSize)
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
	ErrorCode Convolution_CPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

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
		if(this->layerData.lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->layerData.lppNeuron[0].size() != this->filterSize * this->layerData.inputDataStruct.ch)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 入力バッファ保存用のアドレス配列を作成
		this->m_lppInputBuffer.resize(batchSize, NULL);

		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum * this->outputBufferCount];
		}
		
		// パディング後の入力バッファを作成
		this->paddingInputDataStruct.x  = this->layerData.convolutionCountVec.x + this->layerData.layerStructure.FilterSize.x - 1;
		this->paddingInputDataStruct.y  = this->layerData.convolutionCountVec.y + this->layerData.layerStructure.FilterSize.y - 1;
		this->paddingInputDataStruct.z  = this->layerData.convolutionCountVec.z + this->layerData.layerStructure.FilterSize.z - 1;
		this->paddingInputDataStruct.ch = this->layerData.inputDataStruct.ch;
		this->lpPaddingInputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpPaddingInputBuffer[batchNum].resize(this->paddingInputDataStruct.GetDataCount(), 0.0f);
		}


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Convolution_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		// ニューロン/バイアスの誤差を一時保存するバッファを作成
		if(lpDBias.empty() || lppDNeuron.empty())
		{
			this->lpDBias.resize(this->neuronCount);
			this->lppDNeuron.resize(this->neuronCount);
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				this->lppDNeuron[neuronNum].resize(this->filterSize * this->layerData.inputDataStruct.ch);
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
	ErrorCode Convolution_CPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Convolution_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];
		
		// 畳みこみ結合処理
		for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 neuronNum=0; neuronNum<(U32)this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(S32 convZ=0; convZ<this->layerData.convolutionCountVec.z; convZ++)
				{
					for(S32 convY=0; convY<this->layerData.convolutionCountVec.y; convY++)
					{
						for(S32 convX=0; convX<this->layerData.convolutionCountVec.x; convX++)
						{
							U32 outputOffset = POSITION_TO_OFFSET_VECTOR(convX,convY,convZ,neuronNum, this->layerData.convolutionCountVec, this->layerData.layerStructure.Output_Channel);

							// 一時保存用のバッファを作成
							F32 tmp = 0.0f;

							// フィルタを処理する
							for(U32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)(convZ * this->layerData.layerStructure.Stride.z + filterZ - this->layerData.layerStructure.Padding.z);
									if((U32)inputZ >= this->layerData.inputDataStruct.z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)(convY * this->layerData.layerStructure.Stride.y + filterY - this->layerData.layerStructure.Padding.y);
										if((U32)inputY >= this->layerData.inputDataStruct.y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)(convX * this->layerData.layerStructure.Stride.x + filterX - this->layerData.layerStructure.Padding.x);
											if((U32)inputX >= this->layerData.inputDataStruct.x)
												continue;

											const S32 inputOffset  = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, chNum, this->layerData.inputDataStruct);
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize, this->layerData.inputDataStruct.ch);

											tmp += this->layerData.lppNeuron[neuronNum][filterOffset] * this->m_lppInputBuffer[batchNum][inputOffset];
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


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Convolution_CPU::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode Convolution_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Convolution_CPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 出力誤差バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
		
		// 入力誤差バッファのアドレスを配列に格納
		this->m_lpDInputBuffer = o_lppDInputBuffer;
		if(o_lppDInputBuffer)
		{
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				this->m_lppDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
			
			// 入力誤差バッファを初期化
			memset(this->m_lpDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->batchSize);
		}

		
		// ニューロンとバイアスの変化量を初期化
		for(S32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
		{
			this->lpDBias[neuronNum] = 0.0f;
			memset(&this->lppDNeuron[neuronNum][0], 0, this->filterSize * this->layerData.inputDataStruct.ch * sizeof(F32));
		}

		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			// 入力誤差計算
			for(S32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(S32 convZ=0; convZ<this->layerData.convolutionCountVec.z; convZ++)
				{
					for(S32 convY=0; convY<this->layerData.convolutionCountVec.y; convY++)
					{
						for(S32 convX=0; convX<this->layerData.convolutionCountVec.x; convX++)
						{
							U32 outputOffet = POSITION_TO_OFFSET_VECTOR(convX,convY,convZ,neuronNum, this->layerData.convolutionCountVec, this->layerData.layerStructure.Output_Channel);
							F32 dOutput = this->m_lppDOutputBuffer[batchNum][outputOffet];

							// フィルタを処理する
							for(U32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)((convZ * this->layerData.layerStructure.Stride.z) - this->layerData.layerStructure.Padding.z + filterZ);
									if((U32)inputZ>=this->layerData.inputDataStruct.z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)((convY * this->layerData.layerStructure.Stride.y) - this->layerData.layerStructure.Padding.y + filterY);
										if((U32)inputY>=this->layerData.inputDataStruct.y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)((convX * this->layerData.layerStructure.Stride.x) - this->layerData.layerStructure.Padding.x + filterX);
											if((U32)inputX>=this->layerData.inputDataStruct.x)
												continue;

											const S32 inputOffset  = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, chNum, this->layerData.inputDataStruct);
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize, this->layerData.inputDataStruct.ch);

											if(this->m_lpDInputBuffer)
												this->m_lppDInputBuffer[batchNum][inputOffset] += this->layerData.lppNeuron[neuronNum][filterOffset] * dOutput;

											// ニューロンの重み変化量を追加
											this->lppDNeuron[neuronNum][filterOffset] += this->m_lppInputBuffer[batchNum][inputOffset] * dOutput;
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
	ErrorCode Convolution_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		Gravisbell::ErrorCode errCode = this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// 学習差分の反映
		U32 filterDataSize = this->layerData.layerStructure.FilterSize.x * this->layerData.layerStructure.FilterSize.y * this->layerData.layerStructure.FilterSize.z * this->layerData.inputDataStruct.ch;

		for(S32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
		{
			// バイアス更新
			this->layerData.lpBias[neuronNum] += this->lpDBias[neuronNum] * this->learnData.LearnCoeff;

			// 各ニューロンを更新
			for(U32 filterOffset=0; filterOffset<filterDataSize; filterOffset++)
			{
				this->layerData.lppNeuron[neuronNum][filterOffset] += this->lppDNeuron[neuronNum][filterOffset] * this->learnData.LearnCoeff;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Convolution_CPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer;
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode Convolution_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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
