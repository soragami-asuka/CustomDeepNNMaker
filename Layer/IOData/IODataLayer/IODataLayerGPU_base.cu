//===================================
// 入出力データを管理するクラス
// GPU制御
//===================================


#include "stdafx.h"
#include "IODataLayerGPU_base.cuh"


#include<vector>
#include<list>
#include<algorithm>

// UUID関連用
#include<boost/uuid/uuid_generators.hpp>


#define BLOCK_SIZE	(16)

using namespace Gravisbell;

namespace
{
	/** ベクトルの要素同士の掛け算. */
	__global__ void cuda_func_calculateError(const F32* i_lpOutputBuffer, const F32* i_lpTeachBuffer, F32* o_lpErrorMax, F32* o_lpErrorAve, F32* o_lpErrorAve2, F32* o_lpErrorCrossEntropy, U32 i_bachNum, U32 i_bufferSize)
	{
		const U32 inputNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(inputNum >= i_bufferSize)	// 分岐するが末尾のwarpだけなので、処理速度に影響はないはず...
			return;

		const U32 bufferPos = i_bachNum * i_bufferSize + inputNum;

		F32 teach = i_lpTeachBuffer[bufferPos];
		F32 output = i_lpOutputBuffer[bufferPos];

		F32 error = (teach - output);
		F32 error_abs = abs(error);

		F32 crossEntropy = -(F32)(
			      teach  * log(max(0.0001,  output)) +
				 (1 - teach) * log(max(0.0001,1-output))
				 );

		// 誤差を保存
		o_lpErrorMax[inputNum]  = max(o_lpErrorMax[inputNum], error_abs);
		o_lpErrorAve[inputNum]  += error_abs;
		o_lpErrorAve2[inputNum] += error_abs * error_abs;
		o_lpErrorCrossEntropy[inputNum] += crossEntropy;
	}
}



namespace Gravisbell {
namespace Layer {
namespace IOData {

	/** コンストラクタ */
	IODataLayerGPU_base::IODataLayerGPU_base(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
		:	guid				(guid)
		,	ioDataStruct		(ioDataStruct)
		,	lpBatchDataNoList	(NULL)
		,	calcErrorCount		(0)
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	IODataLayerGPU_base::~IODataLayerGPU_base()
	{
		cublasDestroy(cublasHandle);
	}


	//===========================
	// 初期化
	//===========================
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode IODataLayerGPU_base::Initialize(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	//==============================
	// レイヤー共通系
	//==============================
	/** レイヤー種別の取得 */
	U32 IODataLayerGPU_base::GetLayerKind()const
	{
		return ELayerKind::LAYER_KIND_GPU | ELayerKind::LAYER_KIND_SINGLE_INPUT | ELayerKind::LAYER_KIND_SINGLE_OUTPUT | ELayerKind::LAYER_KIND_DATA;
	}

	/** レイヤー固有のGUIDを取得する */
	Gravisbell::GUID IODataLayerGPU_base::GetGUID(void)const
	{
		return this->guid;
	}

	/** レイヤー種別識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	Gravisbell::GUID IODataLayerGPU_base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		Gravisbell::Layer::IOData::GetLayerCode(layerCode);

		return layerCode;
	}
	
	/** レイヤーの設定情報を取得する */
	const SettingData::Standard::IData* IODataLayerGPU_base::GetLayerStructure()const
	{
		return NULL;
	}

	//==============================
	// データ管理系
	//==============================
	/** データの構造情報を取得する */
	IODataStruct IODataLayerGPU_base::GetDataStruct()const
	{
		return this->ioDataStruct;
	}

	/** データのバッファサイズを取得する.
		@return データのバッファサイズ.使用するF32型配列の要素数. */
	U32 IODataLayerGPU_base::GetBufferCount()const
	{
		return this->ioDataStruct.GetDataCount();
	}


	//==============================
	// レイヤー共通系
	//==============================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	Gravisbell::ErrorCode IODataLayerGPU_base::PreProcessLearn(U32 batchSize)
	{
		// 通常の演算用の処理を実行
		ErrorCode err = PreProcessCalculate(batchSize);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 誤差差分データ配列の初期化
		this->lpDInputBuffer.resize(batchSize * this->GetBufferCount());

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	Gravisbell::ErrorCode IODataLayerGPU_base::PreProcessCalculate(U32 batchSize)
	{
		// バッチサイズの保存
		this->batchSize = batchSize;

		// バッファの確保とバッチ処理データ配列の初期化
		this->lpOutputBuffer.resize(batchSize * this->GetBufferCount());

		// 誤差計算用のバッファを初期化
		this->lpErrorValue_max.resize(this->GetBufferCount());
		this->lpErrorValue_ave.resize(this->GetBufferCount());
		this->lpErrorValue_ave2.resize(this->GetBufferCount());
		this->lpErrorValue_crossEntropy.resize(this->GetBufferCount());


		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	Gravisbell::ErrorCode IODataLayerGPU_base::PreProcessLearnLoop(const SettingData::Standard::IData& config)
	{
		return this->PreProcessCalculateLoop();
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	Gravisbell::ErrorCode IODataLayerGPU_base::PreProcessCalculateLoop()
	{
		this->calcErrorCount = 0;
		cudaMemset(thrust::raw_pointer_cast(&this->lpErrorValue_max[0]),			0, sizeof(F32)*this->lpErrorValue_max.size());
		cudaMemset(thrust::raw_pointer_cast(&this->lpErrorValue_ave[0]),			0, sizeof(F32)*this->lpErrorValue_max.size());
		cudaMemset(thrust::raw_pointer_cast(&this->lpErrorValue_ave2[0]),			0, sizeof(F32)*this->lpErrorValue_max.size());
		cudaMemset(thrust::raw_pointer_cast(&this->lpErrorValue_crossEntropy[0]),	0, sizeof(F32)*this->lpErrorValue_max.size());

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** バッチサイズを取得する.
		@return 同時に演算を行うバッチのサイズ */
	U32 IODataLayerGPU_base::GetBatchSize()const
	{
		return this->batchSize;
	}


	//==============================
	// 入力系
	//==============================
	/** 学習誤差を計算する.
		@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要 */
	Gravisbell::ErrorCode IODataLayerGPU_base::CalculateLearnError(Gravisbell::CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		U32 inputBufferCount = this->GetInputBufferCount();

		if(this->lpDInputBuffer.size())
		{
			// データをコピー
			this->lpDInputBuffer = this->lpOutputBuffer;

			// データの誤差を計算
			{
				float alpha = -1.0f;

				// y = alphat * x + y;
				cublasSaxpy(
					this->cublasHandle,
					inputBufferCount * this->batchSize,
					&alpha,
					i_lppInputBuffer,
					1,
					thrust::raw_pointer_cast(&this->lpDInputBuffer[0]),
					1);
			}
		}


		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			U32 bufferCount = this->GetBufferCount();
			dim3 grid((bufferCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
			dim3 block(BLOCK_SIZE, 1, 1);

			cuda_func_calculateError<<<grid, block>>>(
				i_lppInputBuffer,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
				thrust::raw_pointer_cast(&this->lpErrorValue_max[0]),
				thrust::raw_pointer_cast(&this->lpErrorValue_ave[0]),
				thrust::raw_pointer_cast(&this->lpErrorValue_ave2[0]),
				thrust::raw_pointer_cast(&this->lpErrorValue_crossEntropy[0]),
				batchNum,
				this->GetBufferCount());

			this->calcErrorCount++;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 誤差の値を取得する.
		CalculateLearnError()を1回以上実行していない場合、正常に動作しない.
		@param	o_min	最小誤差.
		@param	o_max	最大誤差.
		@param	o_ave	平均誤差.
		@param	o_ave2	平均二乗誤差. */
	ErrorCode IODataLayerGPU_base::GetCalculateErrorValue(F32& o_max, F32& o_ave, F32& o_ave2, F32& o_crossEntropy)
	{
		o_max  = 0.0f;
		o_ave  = 0.0f;
		o_ave2 = 0.0f;
		o_crossEntropy = 0.0f;

		for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
		{
			F32 errorValue_max  = this->lpErrorValue_max[inputNum];
			F32 errorValue_ave  = this->lpErrorValue_ave[inputNum];
			F32 errorValue_ave2 = this->lpErrorValue_ave2[inputNum];
			F32 errorValue_crossEntropy = this->lpErrorValue_crossEntropy[inputNum];

			o_max   = max(o_max, errorValue_max);
			o_ave  += errorValue_ave;
			o_ave2 += errorValue_ave2;
			o_crossEntropy += errorValue_crossEntropy;
		}

		o_ave  = o_ave / this->calcErrorCount / this->GetBufferCount();
		o_ave2 = (F32)sqrt(o_ave2 / this->calcErrorCount / this->GetBufferCount());
		o_crossEntropy = o_crossEntropy / this->calcErrorCount / this->GetBufferCount();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 詳細な誤差の値を取得する.
		各入出力の値毎に誤差を取る.
		CalculateLearnError()を1回以上実行していない場合、正常に動作しない.
		各配列の要素数は[GetBufferCount()]以上である必要がある.
		@param	o_lpMin		最小誤差.
		@param	o_lpMax		最大誤差.
		@param	o_lpAve		平均誤差.
		@param	o_lpAve2	平均二乗誤差. */
	ErrorCode IODataLayerGPU_base::GetCalculateErrorValueDetail(F32 o_lpMax[], F32 o_lpAve[], F32 o_lpAve2[])
	{
		for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
		{
			o_lpMax[inputNum]   = this->lpErrorValue_max[inputNum];
			o_lpAve[inputNum]  += this->lpErrorValue_ave[inputNum] / this->GetDataCount();
			o_lpAve2[inputNum] += (F32)sqrt(this->lpErrorValue_ave2[inputNum] / this->GetDataCount());
		}
			
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 入力データ構造を取得する.
		@return	入力データ構造 */
	IODataStruct IODataLayerGPU_base::GetInputDataStruct()const
	{
		return this->GetDataStruct();
	}

	/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
	U32 IODataLayerGPU_base::GetInputBufferCount()const
	{
		return this->GetBufferCount();
	}

	/** 学習差分を取得する.
		配列の要素数はGetInputBufferCountの戻り値.
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER IODataLayerGPU_base::GetDInputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpDInputBuffer[0]);
	}
	/** 学習差分を取得する.
		@param lpDOutputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	Gravisbell::ErrorCode IODataLayerGPU_base::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*batchSize*inputBufferCount, cudaMemcpyDeviceToHost);

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	//==============================
	// 出力系
	//==============================
	/** 出力データ構造を取得する */
	IODataStruct IODataLayerGPU_base::GetOutputDataStruct()const
	{
		return this->GetDataStruct();
	}

	/** 出力バッファ数を取得する. byte数では無くデータの数なので注意 */
	U32 IODataLayerGPU_base::GetOutputBufferCount()const
	{
		return this->GetBufferCount();
	}

	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER IODataLayerGPU_base::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	Gravisbell::ErrorCode IODataLayerGPU_base::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*batchSize*outputBufferCount, cudaMemcpyDeviceToHost);

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


}	// IOData
}	// Layer
}	// Gravisbell
