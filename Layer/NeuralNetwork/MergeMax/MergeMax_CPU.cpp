//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"MergeMax_DATA.hpp"
#include"MergeMax_FUNC.hpp"
#include"MergeMax_Base.h"

#include"MergeMax_CPU.h"
#include"MergeMax_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	MergeMax_CPU::MergeMax_CPU(Gravisbell::GUID guid, MergeMax_LayerData_CPU& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	MergeMax_Base					(guid, i_lpInputDataStruct, i_layerData.GetOutputDataStruct(&i_lpInputDataStruct[0], (U32)i_lpInputDataStruct.size()))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
	{
	}
	/** デストラクタ */
	MergeMax_CPU::~MergeMax_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 MergeMax_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode MergeMax_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	MergeMax_LayerData_Base& MergeMax_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const MergeMax_LayerData_Base& MergeMax_CPU::GetLayerData()const
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
	ErrorCode MergeMax_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 出力誤差バッファ受け取り用のアドレス配列を作成する
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());

		// 入力差分バッファ受け取り用のアドレス配列を作成する
		this->lppBatchDInputBuffer.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			this->lppBatchDInputBuffer[inputNum].resize(this->GetBatchSize());
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode MergeMax_CPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->lpInputBufferCount.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			this->lpInputBufferCount[inputNum] = this->GetInputBufferCount(inputNum);
		}

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 入力バッファ保存用のアドレス配列を作成
		this->lppBatchInputBuffer.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			this->lppBatchInputBuffer[inputNum].resize(this->GetBatchSize(), NULL);
		}

		// 出力バッファを作成
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode MergeMax_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode MergeMax_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力バッファのアドレスを配列に格納
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->lppBatchInputBuffer[inputNum][batchNum] = &i_lppInputBuffer[inputNum][batchNum * this->lpInputBufferCount[inputNum]];
		}
		// 出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];

		// 出力バッファを-FLT_MAXでクリア
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
			{
				this->lppBatchOutputBuffer[batchNum][bufNum] = -FLT_MAX;
			}
		}

		// 最大値を計算
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
			{
				U32 bufferSize = min(this->lpInputBufferCount[inputNum], outputBufferCount);

				for(U32 bufNum=0; bufNum<bufferSize; bufNum++)
				{
					this->lppBatchOutputBuffer[batchNum][bufNum] = max(this->lppBatchOutputBuffer[batchNum][bufNum], this->lppBatchInputBuffer[inputNum][batchNum][bufNum]);
				}
			}
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
	ErrorCode MergeMax_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 出力誤差バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];

		if(o_lppDInputBuffer)
		{
			// 入力誤差バッファのアドレスを配列に格納
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					this->lppBatchDInputBuffer[inputNum][batchNum] = &o_lppDInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]];
				}

				// バッファのクリア
				memset(o_lppDInputBuffer[inputNum], 0, sizeof(F32)*this->lpInputBufferCount[inputNum]*this->GetBatchSize());
			}
			// 入力バッファのアドレスを配列に格納
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					this->lppBatchInputBuffer[inputNum][batchNum] = &i_lppInputBuffer[inputNum][batchNum * this->lpInputBufferCount[inputNum]];
			}
			// 出力バッファのアドレスを配列に格納
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->lppBatchOutputBuffer[batchNum] = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);

			U32 CH_SIZE = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				U32 offset_output = 0;
				for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
				{
					U32 bufferSize = min(this->lpInputBufferCount[inputNum], outputBufferCount);

					for(U32 bufNum=0; bufNum<bufferSize; bufNum++)
					{
						if(this->lppBatchOutputBuffer[batchNum][bufNum] == this->lppBatchInputBuffer[inputNum][batchNum][bufNum])	// 入力と出力が一致する場合
						{
							this->lppBatchDInputBuffer[inputNum][batchNum][bufNum] = this->lppBatchDOutputBuffer[batchNum][bufNum];
						}
					}
				}
			}
		}

#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer0(this->GetBatchSize() * this->lpInputBufferCount[0]);
		memcpy(&lpTmpInputBuffer0[0], i_lppInputBuffer[0], sizeof(float)*lpTmpInputBuffer0.size());
		std::vector<float> lpTmpInputBuffer1(this->GetBatchSize() * this->lpInputBufferCount[1]);
		memcpy(&lpTmpInputBuffer1[0], i_lppInputBuffer[1], sizeof(float)*lpTmpInputBuffer1.size());

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		memcpy(&lpTmpOutputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size());

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		memcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size());

		std::vector<float> lpTmpDInputBuffer0(this->GetBatchSize() * this->lpInputBufferCount[0]);
		memcpy(&lpTmpDInputBuffer0[0], o_lppDInputBuffer[0], sizeof(float)*lpTmpDInputBuffer0.size());
		std::vector<float> lpTmpDInputBuffer1(this->GetBatchSize() * this->lpInputBufferCount[1]);
		memcpy(&lpTmpDInputBuffer1[0], o_lppDInputBuffer[1], sizeof(float)*lpTmpDInputBuffer1.size());
#endif


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode MergeMax_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
