//======================================
// 出力信号分割レイヤー
// CPU処理用
//======================================
#include"stdafx.h"

#include"Reshape_SquaresCenterCross_DATA.hpp"
#include"Reshape_SquaresCenterCross_FUNC.hpp"
#include"Reshape_SquaresCenterCross_Base.h"

#include"Reshape_SquaresCenterCross_CPU.h"
#include"Reshape_SquaresCenterCross_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Reshape_SquaresCenterCross_CPU::Reshape_SquaresCenterCross_CPU(Gravisbell::GUID guid, Reshape_SquaresCenterCross_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Reshape_SquaresCenterCross_Base		(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData					(i_layerData)	/**< レイヤーデータ */
	{
	}
	/** デストラクタ */
	Reshape_SquaresCenterCross_CPU::~Reshape_SquaresCenterCross_CPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Reshape_SquaresCenterCross_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Reshape_SquaresCenterCross_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Reshape_SquaresCenterCross_LayerData_Base& Reshape_SquaresCenterCross_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const Reshape_SquaresCenterCross_LayerData_Base& Reshape_SquaresCenterCross_CPU::GetLayerData()const
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
	ErrorCode Reshape_SquaresCenterCross_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 入力誤差/出力誤差バッファ受け取り用のアドレス配列を作成する
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());
		this->lppBatchDInputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Reshape_SquaresCenterCross_CPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 入力/出力バッファ保存用のアドレス配列を作成
		this->lppBatchInputBuffer.resize(this->GetBatchSize(), NULL);
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());

		// 変換テーブルを作成
		U32 sqX = sqrt(this->GetInputDataStruct().x-1);
		this->m_lpConvertTable.resize((sqX*2+1) * (sqX*2+1));
		this->m_lppConvertTable.resize(sqX*2+1);
		for(U32 y=0; y<m_lppConvertTable.size(); y++)
		{
			this->m_lppConvertTable[y] = &this->m_lpConvertTable[(sqX*2+1)*y];

			for(U32 x=0; x<m_lppConvertTable.size(); x++)
			{
				U32 tmpX = abs((S32)x-(S32)sqX);
				U32 tmpY = abs((S32)y-(S32)sqX);
				U32 value = tmpY*tmpX;

				this->m_lppConvertTable[y][x] = value;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Reshape_SquaresCenterCross_CPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Reshape_SquaresCenterCross_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力/出力バッファのアドレスを配列に格納
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

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

							this->lppBatchOutputBuffer[batchNum][outputOffset] = this->lppBatchInputBuffer[batchNum][inputOffset];
						}
					}
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
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列の[GetOutputDataCount()]配列
		直前の計算結果を使用する */
	ErrorCode Reshape_SquaresCenterCross_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(i_lppDOutputBuffer && o_lppDInputBuffer)
		{
			// 入力誤差/出力誤差バッファのアドレスを配列に格納
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
				this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
			}

			// 入力誤差を初期化
			memset(o_lppDInputBuffer, 0, sizeof(F32)*this->GetBatchSize()*this->inputBufferCount);

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

								this->lppBatchDInputBuffer[batchNum][inputOffset] += this->lppBatchDOutputBuffer[batchNum][outputOffset];
							}
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
	ErrorCode Reshape_SquaresCenterCross_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
