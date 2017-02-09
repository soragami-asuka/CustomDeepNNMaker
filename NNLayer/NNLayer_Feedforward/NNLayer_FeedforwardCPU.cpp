//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward.h"
#include"NNLayer_FeedforwardBase.h"


class NNLayer_FeedforwardCPU : public NNLayer_FeedforwardBase
{
private:
	std::vector<std::vector<NEURON_TYPE>>	lppNeuron;			/**< 各ニューロンの係数<ニューロン数, 入力数> */
	std::vector<NEURON_TYPE>				lpBias;				/**< ニューロンのバイアス<ニューロン数> */
	std::vector<float>						lpOutput;			/**< 出力バッファ */
	std::vector<float>						lpDInputBuffer;		/**< 入力誤差差分<入力信号数> */
	std::vector<float>						lpDOutputBuffer;	/**< 出力誤差差分<ニューロン数> */

	float learnCoeff;	/**< 学習係数 */

	// Get関数を使うと処理不可がかさむので一時保存用. PreCalculateで値を格納.
	unsigned int inputBufferCount;				/**< 入力バッファ数 */
	unsigned int neuronCount;					/**< ニューロン数 */
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
	ELayerKind GetLayerKind()const
	{
		return LAYER_KIND_CPU_CALC;
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ELayerErrorCode Initialize(void)
	{
		// 入力バッファ数を確認
		unsigned int inputBufferCount = this->GetInputBufferCount();
		if(inputBufferCount == 0)
			return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;

		// ニューロン数を確認
		unsigned int neuronCount = this->GetNeuronCount();
		if(neuronCount == 0)
			return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;

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

		// 出力バッファを確保
		this->lpOutput.resize(neuronCount);
		// 出力差分バッファの確保
		this->lpDInputBuffer.resize(inputBufferCount);

		return LAYER_ERROR_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ELayerErrorCode Initialize(const INNLayerConfig& config)
	{
		// 設定情報を設定
		ELayerErrorCode err = this->SetLayerConfig(config);
		if(err != LAYER_ERROR_NONE)
			return err;

		// 共通の初期化処理
		return this->Initialize();
	}

	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合0 */
	int WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pConfig == NULL)
			return LAYER_ERROR_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// 設定情報
		writeBufferByte += this->pConfig->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

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
	/** レイヤーを読み込む.
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
	int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
	{
		int readBufferByte = 0;

		// 設定情報を読み込む
		if(this->pConfig != NULL)
			delete this->pConfig;
		this->pConfig = ::CreateLayerConfigFromBuffer(i_lpBuffer, i_bufferSize, readBufferByte);

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


		return readBufferByte;
	}

public:
	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ELayerErrorCode PreCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return LAYER_ERROR_FRAUD_INPUT_COUNT;

		// ニューロン数を確認
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return LAYER_ERROR_FRAUD_OUTPUT_COUNT;

		// ニューロンバッファのサイズ確認
		if(this->lppNeuron.size() != this->neuronCount)
			return LAYER_ERROR_FRAUD_NEURON_COUNT;
		if(this->lppNeuron[0].size() != this->inputBufferCount)
			return LAYER_ERROR_FRAUD_NEURON_COUNT;

		// 出力バッファのサイズを確認
		if(this->lpOutput.size() != this->neuronCount)
			return LAYER_ERROR_FRAUD_OUTPUT_COUNT;

		// 入力差分バッファのサイズを確認
		if(this->lpDInputBuffer.size() != this->inputBufferCount)
			return LAYER_ERROR_FRAUD_OUTPUT_COUNT;

		// 出力差分バッファを確保
		this->lpDOutputBuffer.resize(this->neuronCount);

		// 各出力先レイヤー内での出力差分バッファの位置
		this->lpDOutputBufferPosition.resize(this->GetOutputToLayerCount());
		for(unsigned int layerNum=0; layerNum<this->GetOutputToLayerCount(); layerNum++)
		{
			this->lpDOutputBufferPosition[layerNum] = this->GetOutputToLayerByNum(layerNum)->GetInputBufferPositionByLayer(this);
		}

		return LAYER_ERROR_NONE;
	}

	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ELayerErrorCode Calculate()
	{
		for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		{
			float tmp = 0;
			// ニューロンの値を加算
			unsigned int inputNum = 0;
			for(unsigned int inputLayerNum=0; inputLayerNum<this->GetInputFromLayerCount(); inputLayerNum++)
			{
				auto pInputLayer = this->GetInputFromLayerByNum(inputLayerNum);
				for(unsigned int layerInputNum=0; layerInputNum<pInputLayer->GetOutputBufferCount(); layerInputNum++)
				{
					tmp += this->lppNeuron[neuronNum][inputNum] * pInputLayer->GetOutputBuffer()[layerInputNum];
					inputNum++;
				}
				// バイアスを加算
				tmp += this->lpBias[neuronNum];
			}

			// シグモイド関数を演算
			this->lpOutput[neuronNum] = 1 / (1+exp(-tmp));
		}

		return LAYER_ERROR_NONE;
	}

	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	const float* GetOutputBuffer()const
	{
		return &this->lpOutput[0];
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