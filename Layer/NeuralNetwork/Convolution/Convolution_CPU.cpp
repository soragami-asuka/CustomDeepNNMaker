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

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)	(((((z*ySize)+y)*xSize)+x)*chSize+ch)
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)			POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)


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
		,	m_lppInputBuffer				(NULL)	/**< 演算時の入力データ */
		,	onUseDropOut					(false)
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


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 Convolution_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
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
	ErrorCode Convolution_CPU::PreProcessLearn(unsigned int batchSize)
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

		// パディング後の入力バッファを作成
		this->paddingInputDataStruct.x  = this->layerData.convolutionCount.x + this->layerData.layerStructure.FilterSize.x;
		this->paddingInputDataStruct.y  = this->layerData.convolutionCount.y + this->layerData.layerStructure.FilterSize.y;
		this->paddingInputDataStruct.z  = this->layerData.convolutionCount.z + this->layerData.layerStructure.FilterSize.z;
		this->paddingInputDataStruct.ch = this->layerData.inputDataStruct.ch;
		this->lpPaddingInputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpPaddingInputBuffer[batchNum].resize(this->paddingInputDataStruct.GetDataCount(), 0.0f);
		}

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


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Convolution_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
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
						this->lppDropOutBuffer[neuronNum].resize(this->filterSize * this->layerData.inputDataStruct.ch);
				}

				// バッファに1or0を入力
				// 1 : DropOutしない
				// 0 : DropOutする
				for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					for(U32 inputNum=0; inputNum<this->filterSize * this->layerData.inputDataStruct.ch; inputNum++)
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
	ErrorCode Convolution_CPU::PreProcessCalculateLoop()
	{
		this->onUseDropOut = false;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Convolution_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		this->m_lppInputBuffer = i_lppInputBuffer;

		// パディング後の入力バッファにデータを移す
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 paddingZ=0; paddingZ<this->paddingInputDataStruct.z; paddingZ++)
			{
				S32 inputZ = paddingZ - this->layerData.layerStructure.PaddingM.z;
				for(U32 paddingY=0; paddingY<this->paddingInputDataStruct.y; paddingY++)
				{
					S32 inputY = paddingY - this->layerData.layerStructure.PaddingM.y;
					for(U32 paddingX=0; paddingX<this->paddingInputDataStruct.x; paddingX++)
					{
						S32 inputX = paddingX - this->layerData.layerStructure.PaddingM.x;

						if(inputZ<0 || inputZ>=this->layerData.inputDataStruct.z)
							continue;
						if(inputY<0 || inputY>=this->layerData.inputDataStruct.y)
							continue;
						if(inputX<0 || inputX>=this->layerData.inputDataStruct.x)
							continue;

						S32 paddingOffset = POSITION_TO_OFFSET_STRUCT(paddingX, paddingY, paddingZ, 0, this->paddingInputDataStruct);
						S32 inputOffset   = POSITION_TO_OFFSET_STRUCT(inputX, inputY, inputZ, 0, this->layerData.inputDataStruct);

						for(S32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
						{
							this->lpPaddingInputBuffer[batchNum][paddingOffset + chNum] = i_lppInputBuffer[batchNum][inputOffset + chNum];
						}
					}
				}
			}
		}

		// 畳みこみ結合処理
		for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(S32 convZ=this->layerData.convolutionStart.z; convZ<this->layerData.convolutionCount.z; convZ++)
			{
				for(S32 convY=this->layerData.convolutionStart.y; convY<this->layerData.convolutionCount.y; convY++)
				{
					for(S32 convX=this->layerData.convolutionStart.x; convX<this->layerData.convolutionCount.x; convX++)
					{
						U32 outputOffet = POSITION_TO_OFFSET(convX,convY,convZ,0, this->layerData.convolutionCount.x,this->layerData.convolutionCount.y,this->layerData.convolutionCount.z, this->layerData.layerStructure.Output_Channel);

						// 出力初期化
						for(U32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
						{
							this->lpOutputBuffer[batchNum][outputOffet + neuronNum] = this->layerData.lpBias[neuronNum];
						}

						// フィルタを処理する
						for(U32 filterOffset=0; filterOffset<this->filterSize; filterOffset++)
						{
							S32 filterZ =  filterOffset / (this->layerData.layerStructure.FilterSize.y  * this->layerData.layerStructure.FilterSize.x);
							S32 filterY = (filterOffset /  this->layerData.layerStructure.FilterSize.x) % this->layerData.layerStructure.FilterSize.y;
							S32 filterX =  filterOffset %  this->layerData.layerStructure.FilterSize.x;

							S32 inputZ = (S32)(convZ * this->layerData.layerStructure.Stride.z + filterZ);
							S32 inputY = (S32)(convY * this->layerData.layerStructure.Stride.y + filterY);
							S32 inputX = (S32)(convX * this->layerData.layerStructure.Stride.x + filterX);

							S32 inputOffset = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, 0, this->paddingInputDataStruct);

							for(U32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
							{
								for(S32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
								{
									if(this->onUseDropOut)
										this->lpOutputBuffer[batchNum][outputOffet + neuronNum] += this->layerData.lppNeuron[neuronNum][filterOffset*this->layerData.inputDataStruct.ch + chNum] * this->lpPaddingInputBuffer[batchNum][inputOffset+chNum] * this->lppDropOutBuffer[neuronNum][filterOffset*this->layerData.inputDataStruct.ch + chNum];
									else
										this->lpOutputBuffer[batchNum][outputOffet + neuronNum] += this->layerData.lppNeuron[neuronNum][filterOffset*this->layerData.inputDataStruct.ch + chNum] * this->lpPaddingInputBuffer[batchNum][inputOffset+chNum];
								}
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
	CONST_BATCH_BUFFER_POINTER Convolution_CPU::GetOutputBuffer()const
	{
		return &this->lppBatchOutputBuffer[0];
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
	ErrorCode Convolution_CPU::CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBufferPrev)
	{
		this->m_lppDOutputBuffer = i_lppDOutputBufferPrev;

		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			// 入力誤差バッファを初期化
			memset(&this->lpDInputBuffer[batchNum][0], 0, this->lpDInputBuffer[batchNum].size()*sizeof(F32));

			// 入力誤差計算
			for(S32 convZ=this->layerData.convolutionStart.z; convZ<this->layerData.convolutionCount.z; convZ++)
			{
				for(S32 convY=this->layerData.convolutionStart.y; convY<this->layerData.convolutionCount.y; convY++)
				{
					for(S32 convX=this->layerData.convolutionStart.x; convX<this->layerData.convolutionCount.x; convX++)
					{
						U32 outputOffet = POSITION_TO_OFFSET(convX,convY,convZ,0, this->layerData.convolutionCount.x,this->layerData.convolutionCount.y,this->layerData.convolutionCount.z, this->layerData.layerStructure.Output_Channel);

						// フィルタを処理する
						for(U32 filterOffset=0; filterOffset<this->filterSize; filterOffset++)
						{
							S32 filterZ =  filterOffset / (this->layerData.layerStructure.FilterSize.y  * this->layerData.layerStructure.FilterSize.x);
							S32 filterY = (filterOffset /  this->layerData.layerStructure.FilterSize.x) % this->layerData.layerStructure.FilterSize.y;
							S32 filterX =  filterOffset %  this->layerData.layerStructure.FilterSize.x;

							S32 inputZ = (S32)(convZ * this->layerData.layerStructure.Stride.z + filterZ) - this->layerData.layerStructure.PaddingM.z;
							S32 inputY = (S32)(convY * this->layerData.layerStructure.Stride.y + filterY) - this->layerData.layerStructure.PaddingM.y;
							S32 inputX = (S32)(convX * this->layerData.layerStructure.Stride.x + filterX) - this->layerData.layerStructure.PaddingM.x;

							if(inputZ<0 || inputZ>=this->layerData.inputDataStruct.z)
								continue;
							if(inputY<0 || inputZ>=this->layerData.inputDataStruct.y)
								continue;
							if(inputX<0 || inputZ>=this->layerData.inputDataStruct.x)
								continue;

							S32 inputOffset = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, 0, this->layerData.inputDataStruct);


							for(U32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
							{
								for(S32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
								{
									if(this->onUseDropOut)
										this->lpDInputBuffer[batchNum][inputOffset+chNum] += this->layerData.lppNeuron[neuronNum][filterOffset*this->layerData.inputDataStruct.ch + chNum] * this->m_lppDOutputBuffer[batchNum][outputOffet+neuronNum] * this->lppDropOutBuffer[neuronNum][filterOffset*this->layerData.inputDataStruct.ch + chNum];
									else
										this->lpDInputBuffer[batchNum][inputOffset+chNum] += this->layerData.lppNeuron[neuronNum][filterOffset*this->layerData.inputDataStruct.ch + chNum] * this->m_lppDOutputBuffer[batchNum][outputOffet+neuronNum];
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
	ErrorCode Convolution_CPU::ReflectionLearnError(void)
	{
		for(U32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
		{
			// バイアス更新
			{
				// 対象ニューロンにかかるDOutputを加算
				F32 sumDOutput = 0.0f;
				for(S32 convZ=this->layerData.convolutionStart.x; convZ<this->layerData.convolutionCount.z; convZ++)
				{
					for(S32 convY=this->layerData.convolutionStart.y; convY<this->layerData.convolutionCount.y; convY++)
					{
						for(S32 convX=this->layerData.convolutionStart.x; convX<this->layerData.convolutionCount.x; convX++)
						{
							U32 outputOffset = POSITION_TO_OFFSET(convX,convY,convZ,0, this->layerData.convolutionCount.x,this->layerData.convolutionCount.y,this->layerData.convolutionCount.z, this->layerData.layerStructure.Output_Channel);

							for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
							{
								sumDOutput += this->m_lppDOutputBuffer[batchNum][outputOffset + neuronNum];
							}
						}
					}
				}
				// バイアスを更新
				this->layerData.lpBias[neuronNum] += sumDOutput;
			}

			// 各ニューロンを更新
			for(U32 filterOffset=0; filterOffset<this->filterSize; filterOffset++)
			{
				S32 filterZ =  filterOffset / (this->layerData.layerStructure.FilterSize.y  * this->layerData.layerStructure.FilterSize.x);
				S32 filterY = (filterOffset /  this->layerData.layerStructure.FilterSize.x) % this->layerData.layerStructure.FilterSize.y;
				S32 filterX =  filterOffset %  this->layerData.layerStructure.FilterSize.x;

				for(U32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
				{

					// 対象ニューロンの入力に掛かるDOutputを加算
					F32 sumDOutput = 0.0f;
					for(S32 convZ=this->layerData.convolutionStart.x; convZ<this->layerData.convolutionCount.z; convZ++)
					{
						for(S32 convY=this->layerData.convolutionStart.y; convY<this->layerData.convolutionCount.y; convY++)
						{
							for(S32 convX=this->layerData.convolutionStart.x; convX<this->layerData.convolutionCount.x; convX++)
							{
								S32 inputZ = (S32)(convZ * this->layerData.layerStructure.Stride.z + filterZ);
								S32 inputY = (S32)(convY * this->layerData.layerStructure.Stride.y + filterY);
								S32 inputX = (S32)(convX * this->layerData.layerStructure.Stride.x + filterX);

								U32 outputOffset = POSITION_TO_OFFSET(convX,convY,convZ,0, this->layerData.convolutionCount.x,this->layerData.convolutionCount.y,this->layerData.convolutionCount.z, this->layerData.layerStructure.Output_Channel);
								S32 inputOffset = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, chNum, this->paddingInputDataStruct);

								for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
								{

									sumDOutput += this->lpPaddingInputBuffer[batchNum][inputOffset] * this->m_lppDOutputBuffer[batchNum][outputOffset + neuronNum];
								}
							}
						}
					}

					// 重み更新
					this->layerData.lppNeuron[neuronNum][filterOffset*this->layerData.layerStructure.Output_Channel + neuronNum] = sumDOutput;
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Convolution_CPU::GetDInputBuffer()const
	{
		return &this->lppBatchDInputBuffer[0];
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode Convolution_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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
