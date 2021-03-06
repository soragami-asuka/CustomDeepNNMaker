//======================================
// 畳み込みニューラルネットワークの結合レイヤー
// CPU処理用
//======================================
#include"stdafx.h"

#include"UpConvolution_DATA.hpp"
#include"UpConvolution_FUNC.hpp"
#include"UpConvolution_Base.h"

#include"UpConvolution_CPU.h"
#include"UpConvolution_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)
#define POSITION_TO_OFFSET_VECTOR(inX,inY,inZ,inCh,vector,chSize)	POSITION_TO_OFFSET(inX, inY, inZ, inCh, vector.x,    vector.y,    vector.z,    chSize)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	UpConvolution_CPU::UpConvolution_CPU(Gravisbell::GUID guid, UpConvolution_LayerData_CPU& i_layerData)
		:	UpConvolution_Base	(guid)
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	neuronCount						(0)		/**< ニューロン数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
	{
	}
	/** デストラクタ */
	UpConvolution_CPU::~UpConvolution_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 UpConvolution_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode UpConvolution_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	UpConvolution_LayerData_Base& UpConvolution_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const UpConvolution_LayerData_Base& UpConvolution_CPU::GetLayerData()const
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
	ErrorCode UpConvolution_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 出力誤差バッファ受け取り用のアドレス配列を作成する
		this->m_lppDOutputBuffer.resize(batchSize);

		// 入力差分バッファを作成
		this->lpDInputBuffer.resize(this->batchSize * this->inputBufferCount);
		this->lppBatchDInputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lppBatchDInputBuffer[batchNum] = &this->lpDInputBuffer[batchNum*this->inputBufferCount];
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode UpConvolution_CPU::PreProcessCalculate(unsigned int batchSize)
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
		this->paddingInputDataStruct.x  = this->layerData.convolutionCount.x + this->layerData.layerStructure.FilterSize.x - 1;
		this->paddingInputDataStruct.y  = this->layerData.convolutionCount.y + this->layerData.layerStructure.FilterSize.y - 1;
		this->paddingInputDataStruct.z  = this->layerData.convolutionCount.z + this->layerData.layerStructure.FilterSize.z - 1;
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
	ErrorCode UpConvolution_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
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
	ErrorCode UpConvolution_CPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode UpConvolution_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];

		
		// 畳みこみ結合処理
		for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 neuronNum=0; neuronNum<(U32)this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(S32 convZ=0; convZ<this->layerData.convolutionCount.z; convZ++)
				{
					for(S32 convY=0; convY<this->layerData.convolutionCount.y; convY++)
					{
						for(S32 convX=0; convX<this->layerData.convolutionCount.x; convX++)
						{
							U32 outputOffset = POSITION_TO_OFFSET_VECTOR(convX,convY,convZ,neuronNum, this->layerData.convolutionCount, this->layerData.layerStructure.Output_Channel);

							// 一時保存用のバッファを作成
							F32 tmp = 0.0f;

							// フィルタを処理する
							for(U32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)((convZ * this->layerData.layerStructure.Stride.z + filterZ)/this->layerData.layerStructure.UpScale.z - this->layerData.layerStructure.Padding.z);
									if((U32)inputZ >= this->layerData.inputDataStruct.z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)((convY * this->layerData.layerStructure.Stride.y + filterY)/this->layerData.layerStructure.UpScale.y - this->layerData.layerStructure.Padding.y);
										if((U32)inputY >= this->layerData.inputDataStruct.y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)((convX * this->layerData.layerStructure.Stride.x + filterX)/this->layerData.layerStructure.UpScale.x - this->layerData.layerStructure.Padding.x);
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
	CONST_BATCH_BUFFER_POINTER UpConvolution_CPU::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode UpConvolution_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	ErrorCode UpConvolution_CPU::Training(CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		// 出力誤差バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppDOutputBuffer[batchNum] = &i_lpDOutputBufferPrev[batchNum * this->outputBufferCount];

		// 入力誤差バッファを初期化
		memset(&this->lpDInputBuffer[0], 0, sizeof(F32)*this->lpDInputBuffer.size());
		
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
				for(S32 convZ=0; convZ<this->layerData.convolutionCount.z; convZ++)
				{
					for(S32 convY=0; convY<this->layerData.convolutionCount.y; convY++)
					{
						for(S32 convX=0; convX<this->layerData.convolutionCount.x; convX++)
						{
							U32 outputOffet = POSITION_TO_OFFSET_VECTOR(convX,convY,convZ,neuronNum, this->layerData.convolutionCount, this->layerData.layerStructure.Output_Channel);
							F32 dOutput = this->m_lppDOutputBuffer[batchNum][outputOffet];

							// フィルタを処理する
							for(U32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)((convZ * this->layerData.layerStructure.Stride.z + filterZ)/this->layerData.layerStructure.UpScale.z - this->layerData.layerStructure.Padding.z);
									if((U32)inputZ>=this->layerData.inputDataStruct.z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)((convY * this->layerData.layerStructure.Stride.y + filterY)/this->layerData.layerStructure.UpScale.y - this->layerData.layerStructure.Padding.y);
										if((U32)inputY>=this->layerData.inputDataStruct.y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)((convX * this->layerData.layerStructure.Stride.x + filterX)/this->layerData.layerStructure.UpScale.x - this->layerData.layerStructure.Padding.x);
											if((U32)inputX>=this->layerData.inputDataStruct.x)
												continue;

											const S32 inputOffset  = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, chNum, this->layerData.inputDataStruct);
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize, this->layerData.inputDataStruct.ch);


											this->lppBatchDInputBuffer[batchNum][inputOffset] += this->layerData.lppNeuron[neuronNum][filterOffset] * dOutput;

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
	CONST_BATCH_BUFFER_POINTER UpConvolution_CPU::GetDInputBuffer()const
	{
		return &this->lpDInputBuffer[0];
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode UpConvolution_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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
