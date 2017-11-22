//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"Reshape_SquaresZeroSideLeftTop_DATA.hpp"
#include"Reshape_SquaresZeroSideLeftTop_FUNC.hpp"
#include"Reshape_SquaresZeroSideLeftTop_Base.h"

#include"Reshape_SquaresZeroSideLeftTop_GPU.cuh"
#include"Reshape_SquaresZeroSideLeftTop_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Reshape_SquaresZeroSideLeftTop_GPU::Reshape_SquaresZeroSideLeftTop_GPU(Gravisbell::GUID guid, Reshape_SquaresZeroSideLeftTop_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct)
		:	Reshape_SquaresZeroSideLeftTop_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	m_lpInputBuffer				(NULL)		/**< 演算時の入力データ */
		,	m_lpDOutputBuffer			(NULL)		/**< 出力誤差データ */
	{
	}
	/** デストラクタ */
	Reshape_SquaresZeroSideLeftTop_GPU::~Reshape_SquaresZeroSideLeftTop_GPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Reshape_SquaresZeroSideLeftTop_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Reshape_SquaresZeroSideLeftTop_LayerData_Base& Reshape_SquaresZeroSideLeftTop_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Reshape_SquaresZeroSideLeftTop_LayerData_Base& Reshape_SquaresZeroSideLeftTop_GPU::GetLayerData()const
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
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 出力誤差バッファ
		this->m_lpDOutputBuffer_h.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lppDOutputBuffer.resize(this->GetBatchSize());
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->m_lppDOutputBuffer[batchNum] = &this->m_lpDOutputBuffer_h[batchNum * this->outputBufferCount];

		// 入力誤差バッファ
		this->m_lpDInputBuffer_h.resize(this->GetBatchSize() * this->inputBufferCount);
		this->m_lppDInputBuffer.resize(this->GetBatchSize());
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->m_lppDInputBuffer[batchNum] = &this->m_lpDInputBuffer_h[batchNum * this->inputBufferCount];


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 入力バッファ保存用のアドレス配列を作成
		this->m_lpInputBuffer_h.resize(this->inputBufferCount * this->GetBatchSize());
		this->m_lppInputBuffer.resize(this->GetBatchSize(), NULL);
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum] = &this->m_lpInputBuffer_h[batchNum * this->inputBufferCount];
		}

		// 出力バッファを作成
		this->m_lpOutputBuffer.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lpOutputBuffer_d.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lppOutputBuffer.resize(this->GetBatchSize());
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppOutputBuffer[batchNum] = &this->m_lpOutputBuffer[batchNum * this->outputBufferCount];
		}

		// 変換テーブルを作成
		this->m_lpConvertTable.resize(this->GetOutputDataStruct().y * this->GetOutputDataStruct().x);
		this->m_lppConvertTable.resize(this->GetOutputDataStruct().y);
		for(U32 y=0; y<this->GetOutputDataStruct().y; y++)
		{
			this->m_lppConvertTable[y] = &this->m_lpConvertTable[this->GetOutputDataStruct().x*y];

			for(U32 x=0; x<this->GetOutputDataStruct().x; x++)
			{
				U32 value = x*y;

				this->m_lppConvertTable[y][x] = value;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		this->m_lpInputBuffer = i_lpInputBuffer;
		cudaMemcpy(&this->m_lpInputBuffer_h[0], this->m_lpInputBuffer, sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToHost);

		// 出力バッファに変換
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 ch=0; ch<this->GetOutputDataStruct().ch; ch++)
			{
				for(U32 outputZ=0; outputZ<this->GetOutputDataStruct().z; outputZ++)
				{
					for(U32 outputY=0; outputY<this->GetOutputDataStruct().y; outputY++)
					{
						for(U32 outputX=0; outputX<this->GetOutputDataStruct().x; outputX++)
						{
							U32 inputX = this->m_lppConvertTable[outputY][outputX];

							U32 outputOffset  = this->GetOutputDataStruct().POSITION_TO_OFFSET(outputX, outputY, outputZ, ch);
							U32 inputOffset   = this->GetInputDataStruct().POSITION_TO_OFFSET(inputX,  0, 0, ch);

							this->m_lppOutputBuffer[batchNum][outputOffset] = this->m_lppInputBuffer[batchNum][inputOffset];
						}
					}
				}
			}
		}

		// 出力バッファをデバイスにコピー
		cudaMemcpy(thrust::raw_pointer_cast(&this->m_lpOutputBuffer_d[0]), &this->m_lpOutputBuffer[0], sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyHostToDevice);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Reshape_SquaresZeroSideLeftTop_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->m_lpOutputBuffer_d[0]);
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToHost);

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
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lpDOutputBuffer)
	{
		// 入力誤差計算
		this->m_lpDOutputBuffer = i_lpDOutputBuffer;
		this->m_lpDInputBuffer  = o_lppDInputBuffer;
		if(this->m_lpDOutputBuffer && this->m_lpDInputBuffer)
		{
			// 出力誤差バッファをホストにコピー
			cudaMemcpy(&this->m_lpDOutputBuffer_h[0], i_lpDOutputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToHost);

			// 入力誤差を初期化
			memset(&this->m_lpDInputBuffer_h[0], 0, sizeof(F32)*this->GetBatchSize()*this->inputBufferCount);

			// 入力誤差計算
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 ch=0; ch<this->GetOutputDataStruct().ch; ch++)
				{
					for(U32 outputZ=0; outputZ<this->GetOutputDataStruct().z; outputZ++)
					{
						for(U32 outputY=0; outputY<this->GetOutputDataStruct().y; outputY++)
						{
							for(U32 outputX=0; outputX<this->GetOutputDataStruct().x; outputX++)
							{
								U32 inputX = this->m_lppConvertTable[outputY][outputX];

								U32 outputOffset  = this->GetOutputDataStruct().POSITION_TO_OFFSET(outputX, outputY, outputZ, ch);
								U32 inputOffset   = this->GetInputDataStruct().POSITION_TO_OFFSET(inputX,  0, 0, ch);

								this->m_lppDInputBuffer[batchNum][inputOffset] += this->m_lppDOutputBuffer[batchNum][outputOffset];
							}
						}
					}
				}
			}

			// 入力誤差をデバイスにコピー
			cudaMemcpy(m_lpDInputBuffer, &this->m_lpDInputBuffer_h[0], sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyHostToDevice);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lpDOutputBuffer)
	{
		return this->CalculateDInput(o_lppDInputBuffer, i_lpDOutputBuffer);
	}


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Reshape_SquaresZeroSideLeftTop_GPU::GetDInputBuffer()const
	{
		return this->m_lpDOutputBuffer;
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*inputBufferCount*batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
