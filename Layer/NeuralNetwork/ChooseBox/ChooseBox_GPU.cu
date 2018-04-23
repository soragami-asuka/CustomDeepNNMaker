//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"ChooseBox_DATA.hpp"
#include"ChooseBox_FUNC.hpp"
#include"ChooseBox_Base.h"

#include"ChooseBox_GPU.cuh"
#include"ChooseBox_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

#define CALC_BATCH_MAX	(256)
#define CALC_INPUT_MAX	(1024)

	__global__ void device_ChooseBox(
		U32 chCount,
		U32 startX, U32 startY, U32 startZ,
		U32 inputXCount,  U32 inputYCount,  U32 inputZCount,
		U32 outputXCount, U32 outputYCount, U32 outputZCount,
		const F32 lpInputBuffer[],
		F32 lpOutputBuffer[])
	{
		U32 batchNum = blockIdx.y;
		U32 ch = blockIdx.x;
		U32 x = threadIdx.x;
		U32 y = threadIdx.y;
		U32 z = threadIdx.z;
		U32 inputX = startX + x;
		U32 inputY = startY + y;
		U32 inputZ = startZ + z;

		U32 inputOffset  = CalculateOffset(batchNum, chCount, inputXCount,  inputYCount,  inputZCount,  ch, inputX, inputY, inputZ);
		U32 outputOffset = CalculateOffset(batchNum, chCount, outputXCount, outputYCount, outputZCount, ch, x, y, z);

		lpOutputBuffer[outputOffset] = lpInputBuffer[inputOffset];
	}
	
	__global__ void device_ReChooseBox(
		U32 chCount,
		U32 startX, U32 startY, U32 startZ,
		U32 inputXCount,  U32 inputYCount,  U32 inputZCount,
		U32 outputXCount, U32 outputYCount, U32 outputZCount,
		const F32 lpDOutputBuffer[],
		F32 lpDInputBuffer[])
	{
		U32 batchNum = blockIdx.y;
		U32 ch = blockIdx.x;
		U32 x = threadIdx.x;
		U32 y = threadIdx.y;
		U32 z = threadIdx.z;
		U32 inputX = startX + x;
		U32 inputY = startY + y;
		U32 inputZ = startZ + z;

		U32 inputOffset  = CalculateOffset(batchNum, chCount, inputXCount,  inputYCount,  inputZCount,  ch, inputX, inputY, inputZ);
		U32 outputOffset = CalculateOffset(batchNum, chCount, outputXCount, outputYCount, outputZCount, ch, x, y, z);

		lpDInputBuffer[inputOffset] = lpDOutputBuffer[outputOffset];
	}

	/** コンストラクタ */
	ChooseBox_GPU::ChooseBox_GPU(Gravisbell::GUID guid, ChooseBox_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	ChooseBox_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	ChooseBox_GPU::~ChooseBox_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 ChooseBox_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode ChooseBox_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	ChooseBox_LayerData_Base& ChooseBox_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const ChooseBox_LayerData_Base& ChooseBox_GPU::GetLayerData()const
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
	ErrorCode ChooseBox_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode ChooseBox_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode ChooseBox_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode ChooseBox_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 出力バッファの初期化
		cudaMemset(o_lppOutputBuffer, 0, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());

		dim3 grid(
			this->GetOutputDataStruct().ch,
			this->GetBatchSize());
		dim3 block(
			this->layerData.layerStructure.boxSize.x,
			this->layerData.layerStructure.boxSize.y,
			this->layerData.layerStructure.boxSize.z);

		device_ChooseBox<<<grid, block>>>(
			this->GetOutputDataStruct().ch,
			this->layerData.layerStructure.startPosition.x, this->layerData.layerStructure.startPosition.y, this->layerData.layerStructure.startPosition.z,
			this->GetInputDataStruct().x, this->GetInputDataStruct().y, this->GetInputDataStruct().z,
			this->layerData.layerStructure.boxSize.x, this->layerData.layerStructure.boxSize.y, this->layerData.layerStructure.boxSize.z,
			i_lppInputBuffer,
			o_lppOutputBuffer);

#if _DEBUG
			std::vector<F32> lpInputBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpInputBuffer[0], i_lppInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpOutputBuffer[0], o_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

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
	ErrorCode ChooseBox_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 出力バッファの初期化
			cudaMemset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

			dim3 grid(
				this->GetOutputDataStruct().ch,
				this->GetBatchSize());
			dim3 block(
				this->layerData.layerStructure.boxSize.x,
				this->layerData.layerStructure.boxSize.y,
				this->layerData.layerStructure.boxSize.z);

			device_ReChooseBox<<<grid, block>>>(
				this->GetOutputDataStruct().ch,
				this->layerData.layerStructure.startPosition.x, this->layerData.layerStructure.startPosition.y, this->layerData.layerStructure.startPosition.z,
				this->GetInputDataStruct().x, this->GetInputDataStruct().y, this->GetInputDataStruct().z,
				this->layerData.layerStructure.boxSize.x, this->layerData.layerStructure.boxSize.y, this->layerData.layerStructure.boxSize.z,
				i_lppDOutputBuffer,
				o_lppDInputBuffer);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode ChooseBox_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
