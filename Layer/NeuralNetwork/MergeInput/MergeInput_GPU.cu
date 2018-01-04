//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"MergeInput_DATA.hpp"
#include"MergeInput_FUNC.hpp"
#include"MergeInput_Base.h"

#include"MergeInput_GPU.cuh"
#include"MergeInput_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	MergeInput_GPU::MergeInput_GPU(Gravisbell::GUID guid, MergeInput_LayerData_GPU& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	MergeInput_Base					(guid, i_lpInputDataStruct, i_layerData.GetOutputDataStruct(&i_lpInputDataStruct[0], (U32)i_lpInputDataStruct.size()))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
	{
	}
	/** デストラクタ */
	MergeInput_GPU::~MergeInput_GPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 MergeInput_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode MergeInput_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	MergeInput_LayerData_Base& MergeInput_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const MergeInput_LayerData_Base& MergeInput_GPU::GetLayerData()const
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
	ErrorCode MergeInput_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 出力誤差バッファの配列を作成
		this->m_lppDInputBuffer.resize(this->GetInputDataCount());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode MergeInput_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->lpInputBufferCount.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
		{
			this->lpInputBufferCount[inputNum] = this->GetInputBufferCount(inputNum);
			if(this->lpInputBufferCount[inputNum] == 0)
				return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		}

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode MergeInput_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode MergeInput_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		switch(this->layerData.layerStructure.mergeDirection)
		{
		case MergeInput::LayerStructure::mergeDirection_ch:
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					U32 offset = 0;
					for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
					{
						cudaError_t err = cudaMemcpyAsync(
							&o_lppOutputBuffer[batchNum*this->outputBufferCount + offset],
							&i_lppInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]],
							sizeof(F32) * this->lpInputBufferCount[inputNum],
							cudaMemcpyDeviceToDevice);
						if(err != 0)
							return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

						offset += this->lpInputBufferCount[inputNum];
					}
				}
				cudaThreadSynchronize();
			}
			break;
		case MergeInput::LayerStructure::mergeDirection_x:
			break;
		case MergeInput::LayerStructure::mergeDirection_y:
			break;
		case MergeInput::LayerStructure::mergeDirection_z:
			break;
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
	ErrorCode MergeInput_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			// 入力誤差バッファのアドレスを配列に格納
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				this->m_lppDInputBuffer[inputNum] = o_lppDInputBuffer[inputNum];
			}

			switch(this->layerData.layerStructure.mergeDirection)
			{
			case MergeInput::LayerStructure::mergeDirection_ch:
				{
					for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					{
						U32 offset = 0;
						for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
						{
							cudaError_t err = cudaMemcpyAsync(
								&this->m_lppDInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]],
								&i_lppDOutputBuffer[batchNum*this->outputBufferCount + offset],
								sizeof(F32) * this->lpInputBufferCount[inputNum],
								cudaMemcpyDeviceToDevice);
							if(err != 0)
								return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

							offset += this->lpInputBufferCount[inputNum];
						}
					}
					cudaThreadSynchronize();
				}
				break;
			case MergeInput::LayerStructure::mergeDirection_x:
				break;
			case MergeInput::LayerStructure::mergeDirection_y:
				break;
			case MergeInput::LayerStructure::mergeDirection_z:
				break;
			}
		}
		

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode MergeInput_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
