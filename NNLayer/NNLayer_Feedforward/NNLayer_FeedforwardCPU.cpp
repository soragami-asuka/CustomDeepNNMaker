//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward_DATA.hpp"
#include"NNLayer_FeedforwardBase.h"
#include"NNLayer_Feedforward_FUNC.hpp"

using namespace Gravisbell;
using namespace Gravisbell::NeuralNetwork;

class NNLayer_FeedforwardCPU : public NNLayer_FeedforwardBase
{
private:
	// 本体
	std::vector<std::vector<NEURON_TYPE>>	lppNeuron;			/**< 各ニューロンの係数<ニューロン数, 入力数> */
	std::vector<NEURON_TYPE>				lpBias;				/**< ニューロンのバイアス<ニューロン数> */

	// 入出力バッファ
	std::vector<std::vector<F32>>						lpOutputBuffer;		/**< 出力バッファ */
	std::vector<std::vector<F32>>						lpDInputBuffer;		/**< 入力誤差差分<入力信号数> */

	float learnCoeff;	/**< 学習係数 */

	// Get関数を使うと処理不可がかさむので一時保存用. PreCalculateで値を格納.
	U32 inputBufferCount;				/**< 入力バッファ数 */
	U32 neuronCount;					/**< ニューロン数 */
	U32 outputBufferCount;				/**< 出力バッファ数 */
	std::vector<int> lpDOutputBufferPosition;	/**< 各出力先レイヤー内での出力差分バッファの位置 */

public:
	/** コンストラクタ */
	NNLayer_FeedforwardCPU(GUID guid)
		:	NNLayer_FeedforwardBase(guid)
		,	learnCoeff	(0.01f)
	{
	}
	/** デストラクタ */
	virtual ~NNLayer_FeedforwardCPU()
	{
	}

public:
	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 GetLayerKind()const
	{
		return LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Initialize(void)
	{
		// 入力バッファ数を確認
		unsigned int inputBufferCount = this->GetInputBufferCount();
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// ニューロン数を確認
		unsigned int neuronCount = this->GetNeuronCount();
		if(neuronCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// バッファを確保しつつ、初期値を設定
		this->lppNeuron.resize(neuronCount);
		this->lpBias.resize(neuronCount);
		for(unsigned int neuronNum=0; neuronNum<lppNeuron.size(); neuronNum++)
		{
			float maxArea = sqrt(6.0f / (0.5f*inputBufferCount + 0.5f*neuronCount));

			// バイアス
			this->lpBias[neuronNum] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * maxArea;

			// ニューロン
			lppNeuron[neuronNum].resize(inputBufferCount);
			for(unsigned int inputNum=0; inputNum<lppNeuron[neuronNum].size(); inputNum++)
			{
				lppNeuron[neuronNum][inputNum] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * maxArea;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
	{
		ErrorCode err;

		// 設定情報の登録
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 入力データ構造の設定
		this->inputDataStruct = i_inputDataStruct;

		this->Initialize();
	}
	/** 初期化. バッファからデータを読み込む
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@return	成功した場合0 */
	ErrorCode InitializeFromBuffer(BYTE* i_lpBuffer, int i_bufferSize)
	{
		int readBufferByte = 0;

		// 設定情報を読み込む
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(i_lpBuffer, i_bufferSize, readBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// 初期化する
		this->Initialize();

		// ニューロン係数
		for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		{
			memcpy(&this->lppNeuron[neuronNum][0], &i_lpBuffer[readBufferByte], this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE));
			readBufferByte += this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE);
		}

		// バイアス
		memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], this->lpBias.size() * sizeof(NEURON_TYPE));
		readBufferByte += this->lpBias.size() * sizeof(NEURON_TYPE);


		return ErrorCode::ERROR_CODE_NONE;
	}

	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合0 */
	int WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// ニューロン係数
		for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		{
			memcpy(&o_lpBuffer[writeBufferByte], &this->lppNeuron[neuronNum][0], this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE));
			writeBufferByte += this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE);
		}

		// バイアス
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], this->lpBias.size() * sizeof(NEURON_TYPE));
		writeBufferByte += this->lpBias.size() * sizeof(NEURON_TYPE);

		return writeBufferByte;
	}

public:
	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode PreProcessLearn(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// ニューロン数を確認
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ニューロンバッファのサイズ確認
		if(this->lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->lppNeuron[0].size() != this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;


		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpOutputBuffer[batchNum].resize(this->outputBufferCount);
		}

		// 入力差分バッファを作成
		this->lpDInputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpDInputBuffer[batchNum].resize(this->inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// ニューロン数を確認
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ニューロンバッファのサイズ確認
		if(this->lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->lppNeuron[0].size() != this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;


		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpOutputBuffer[batchNum].resize(this->outputBufferCount);
		}

		// 入力差分バッファを作成
		// はスキップ

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		// 構造体に読み込む
		this->pLearnData->WriteToStruct((BYTE*)&this->learnData);
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				float tmp = 0;

				// ニューロンの値を加算
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					tmp += i_lppInputBuffer[batchNum][inputNum] * this->lppNeuron[neuronNum][inputNum];
				}
				tmp += this->lpBias[neuronNum];

				// シグモイド関数を演算
				this->lpOutputBuffer[batchNum][neuronNum] = 1 / (1+exp(-tmp));
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 出力データバッファを取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** 出力データバッファを取得する.
		@param lpOutputBuffer	出力データ格納先配列. GetOutputBufferCountで取得した値の要素数が必要
		@return 成功した場合0 */
	ELayerErrorCode GetOutputBuffer(float lpOutputBuffer[])const
	{
		memcpy(lpOutputBuffer, &this->lpOutput[0], this->lpOutput.size() * sizeof(float));

		return LAYER_ERROR_NONE;
	}

public:
	//================================
	// 学習処理
	//================================
	/** 学習誤差を計算する.
		直前の計算結果を使用する */
	ELayerErrorCode CalculateLearnError()
	{
		// 出力誤差差分を計算
		for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// 各出力先レイヤーの出力差分の合計値を求める
			float sumDOutput = 0.0f;
			for(unsigned int outputToLayerNum=0; outputToLayerNum<this->GetOutputToLayerCount(); outputToLayerNum++)
			{
				sumDOutput += this->GetOutputToLayerByNum(outputToLayerNum)->GetDInputBuffer()[this->lpDOutputBufferPosition[outputToLayerNum] + neuronNum];
			}
			this->lpDOutputBuffer[neuronNum] = sumDOutput;
		}

		// 入力誤差差分を計算
		unsigned int inputNum = 0;
		for(unsigned int inputLayerNum=0; inputLayerNum<this->GetInputFromLayerCount(); inputLayerNum++)
		{
			auto pInputLayer = this->GetInputFromLayerByNum(inputLayerNum);
			for(unsigned int layerInputNum=0; layerInputNum<pInputLayer->GetOutputBufferCount(); layerInputNum++)
			{
				float tmp = 0.0f;

				for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					tmp += this->lpDOutputBuffer[neuronNum] * this->lppNeuron[neuronNum][inputNum];
				}

				this->lpDInputBuffer[inputNum] = pInputLayer->GetOutputBuffer()[layerInputNum] * (1.0f - pInputLayer->GetOutputBuffer()[layerInputNum]) * tmp;
				inputNum++;
			}
		}

		return LAYER_ERROR_NONE;
	}

	/** 誤差差分をレイヤーに反映させる */
	ELayerErrorCode ReflectionLearnError(void)
	{
		for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// バイアス更新
			this->lpBias[neuronNum] += this->learnCoeff * this->lpDOutputBuffer[neuronNum];

			// 入力対応ニューロン更新
			unsigned int inputNum = 0;
			for(unsigned int inputLayerNum=0; inputLayerNum<this->GetInputFromLayerCount(); inputLayerNum++)
			{
				auto pInputLayer = this->GetInputFromLayerByNum(inputLayerNum);
				for(unsigned int layerInputNum=0; layerInputNum<pInputLayer->GetOutputBufferCount(); layerInputNum++)
				{
					this->lppNeuron[neuronNum][inputNum] += this->learnCoeff * this->lpDOutputBuffer[neuronNum] * pInputLayer->GetOutputBuffer()[layerInputNum];
					inputNum++;
				}
			}
		}

		return LAYER_ERROR_NONE;
	}

	/** 学習差分を取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return	誤差差分配列の先頭ポインタ */
	const float* GetDInputBuffer()const
	{
		return &this->lpDInputBuffer[0];
	}
	/** 学習差分を取得する.
		@param lpDOutputBuffer	学習差分を格納する配列. GetOutputBufferCountで取得した値の要素数が必要 */
	ELayerErrorCode GetDInputBuffer(float o_lpDInputBuffer[])const
	{
		memcpy(o_lpDInputBuffer, &this->lpDInputBuffer[0], this->lpDInputBuffer.size() * sizeof(float));

		return LAYER_ERROR_NONE;
	}
};


/** CPU処理用のレイヤーを作成 */
EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerCPU(GUID guid)
{
	return new NNLayer_FeedforwardCPU(guid);

	return NULL;
}