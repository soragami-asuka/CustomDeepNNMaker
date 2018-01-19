//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// GPU処理用
//======================================
#include"stdafx.h"

#include"SOM_DATA.hpp"
#include"SOM_FUNC.hpp"
#include"SOM_Base.h"

#include"SOM_GPU.cuh"
#include"SOM_LayerData_GPU.cuh"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define BLOCK_SIZE	(16)

namespace
{
}

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

#define CODE_MATCH_RATE	(L"SOM_MATCH_RATE")

	/** コンストラクタ */
	SOM_GPU::SOM_GPU(Gravisbell::GUID guid, SOM_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	SOM_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	unitCount						(0)		/**< ユニット数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
		,	temporaryMemoryManager			(i_temporaryMemoryManager)
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	SOM_GPU::~SOM_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 SOM_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode SOM_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	SOM_LayerData_Base& SOM_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const SOM_LayerData_Base& SOM_GPU::GetLayerData()const
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
	ErrorCode SOM_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// パラメータの変化量バッファ
		this->lpDUnit.resize(this->unitCount * this->inputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode SOM_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// ニューロン数を確認
		this->unitCount = this->GetUnitCount();
		if(this->unitCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ニューロンバッファのサイズ確認
		if(this->layerData.lpUnitData.size() != this->unitCount * this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;


		// 各ユニットの座標を計算する
		thrust::host_vector<F32> lpTmpUnitPos(this->unitCount * this->layerData.layerStructure.DimensionCount);
		for(U32 unitNo=0; unitNo<this->unitCount; unitNo++)
		{
			U32 offset = unitNo * this->layerData.layerStructure.DimensionCount;

			U32 tmpNo = unitNo;
			for(U32 dimNo=0; dimNo<(U32)this->layerData.layerStructure.DimensionCount; dimNo++)
			{
				U32 pos = tmpNo % this->layerData.layerStructure.ResolutionCount;
				tmpNo /= this->layerData.layerStructure.ResolutionCount;

				lpTmpUnitPos[offset + dimNo] = (F32)pos / this->layerData.layerStructure.ResolutionCount;
			}
		}
		this->lpUnitPos = lpTmpUnitPos;


		// 一時バッファのサイズを設定
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), CODE_MATCH_RATE, sizeof(F32)*this->unitCount*this->layerData.layerStructure.DimensionCount);


		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode SOM_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode SOM_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// BMU(Best Matching Unit)を調べる
		{
			// バッファを確保
			F32* lpTmpMatchRate = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), CODE_MATCH_RATE);

			// 一致率を計算
			{
				// C = aAB + bC;

				// CUBLASは
				// 0, 4,  8
				// 1, 5,  9
				// 2, 6, 10
				// 3, 7, 11
				// のように縦方向にインデックスが進む行列で構成されている


				F32 alpha = 1.0f;
				F32 beta  = 0.0f;

				cublasSgemm(
					this->cublasHandle,
					CUBLAS_OP_T,
					CUBLAS_OP_N,
					this->unitCount,	// 行列Aの行数
					this->GetBatchSize(),	// 行列Bの列数
					this->inputBufferCount,	// 行列Aの列数,行列Bの行数
					&alpha,
					thrust::raw_pointer_cast(&this->layerData.lpUnitData[0]),	// 行列A
					this->inputBufferCount,										// 行列Aの転置前の行数
					i_lppInputBuffer,											// 行列B
					this->inputBufferCount,										// 行列Bの転置前の行数
					&beta,
					&lpTmpMatchRate[0],
					this->outputBufferCount);
			}

			// 最大値を求める
			{
				// 絶対値の最大値を求める

			}

			// ユニット座標を求める

			// バッファ解放
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), CODE_MATCH_RATE);
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
	ErrorCode SOM_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudaMemset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode SOM_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
