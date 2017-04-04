//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"FullyConnect_Activation_DATA.hpp"
#include"FullyConnect_Activation_FUNC.hpp"
#include"FullyConnect_Activation_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

class FeedforwardCPU : public FullyConnect_Activation_Base
{
private:
	// 本体
	std::vector<std::vector<NEURON_TYPE>>	lppNeuron;			/**< 各ニューロンの係数<ニューロン数, 入力数> */
	std::vector<NEURON_TYPE>				lpBias;				/**< ニューロンのバイアス<ニューロン数> */

	// 入出力バッファ
	std::vector<std::vector<F32>>			lpOutputBuffer;		/**< 出力バッファ <バッチ数><ニューロン数> */
	std::vector<std::vector<F32>>			lpDInputBuffer;		/**< 入力誤差差分 <バッチ数><入力信号数> */
	
	std::vector<F32*>						lppBatchOutputBuffer;		/**< バッチ処理用出力バッファ <バッチ数> */
	std::vector<F32*>						lppBatchDInputBuffer;		/**< バッチ処理用入力誤差差分 <バッチ数> */

	// Get関数を使うと処理不可がかさむので一時保存用. PreCalculateで値を格納.
	U32 inputBufferCount;				/**< 入力バッファ数 */
	U32 neuronCount;					/**< ニューロン数 */
	U32 outputBufferCount;				/**< 出力バッファ数 */

	// 演算時の入力データ
	CONST_BATCH_BUFFER_POINTER m_lppInputBuffer;	/**< 演算時の入力データ */
	CONST_BATCH_BUFFER_POINTER m_lppDOutputBuffer;	/**< 入力誤差計算時の出力誤差データ */

	// 演算処理用のバッファ
	bool onUseDropOut;											/**< ドロップアウト処理を実行するフラグ. */
	std::vector<std::vector<NEURON_TYPE>>	lppDropOutBuffer;	/**< ドロップアウト処理用の係数<ニューロン数, 入力数> */

public:
	/** コンストラクタ */
	FeedforwardCPU(Gravisbell::GUID guid)
		:	FullyConnect_Activation_Base	(guid)
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	neuronCount						(0)		/**< ニューロン数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
		,	m_lppInputBuffer				(NULL)	/**< 演算時の入力データ */
		,	m_lppDOutputBuffer				(NULL)	/**< 入力誤差計算時の出力誤差データ */
		,	onUseDropOut					(false)	
	{
	}
	/** デストラクタ */
	virtual ~FeedforwardCPU()
	{
	}

public:
	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
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

		return this->Initialize();
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
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpOutputBuffer[batchNum].resize(this->outputBufferCount);
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum][0];
		}

		// ドロップアウト処理を未使用に変更
		this->onUseDropOut = false;
		this->lppDropOutBuffer.clear();

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

		// ドロップアウト
		{
			auto pItem = dynamic_cast<const Gravisbell::SettingData::Standard::IItem_Float*>(data.GetItemByID(L"DropOut"));
			if(pItem)
				this->learnData.DropOut = pItem->GetValue();
			else
				this->learnData.DropOut = 0.0f;
			
			S32 dropOutRate = (S32)(learnData.DropOut * RAND_MAX);

			if(dropOutRate > 0)
			{
				this->onUseDropOut = true;
				if(this->lppDropOutBuffer.empty())
				{
					// バッファの確保
					this->lppDropOutBuffer.resize(this->neuronCount);
					for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
						this->lppDropOutBuffer[neuronNum].resize(this->inputBufferCount);
				}

				// バッファに1or0を入力
				// 1 : DropOutしない
				// 0 : DropOutする
				for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
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
	ErrorCode PreProcessCalculateLoop()
	{
		this->onUseDropOut = false;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		this->m_lppInputBuffer = i_lppInputBuffer;

		// ループ内でif分を実行すると処理不可がかさむので外で分ける.
		if(!this->onUseDropOut)
		{
			// DropOutを使用しない場合
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

					if(this->learnData.DropOut > 0.0f)
						tmp *= (1.0f - this->learnData.DropOut);

					// 活性化関数
					if(this->layerStructure.ActivationType == Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU)
					{
						// ReLU
						this->lpOutputBuffer[batchNum][neuronNum] = max(0.0f, tmp);
					}
					else
					{
						// シグモイド関数を演算
						this->lpOutputBuffer[batchNum][neuronNum] = 1 / (1+exp(-tmp));
					}
					
					#ifdef _DEBUG
					if(isnan(this->lpOutputBuffer[batchNum][neuronNum]))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
				}
			}
		}
		else
		{
			// DropOutを使用する場合
			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					float tmp = 0;

					// ニューロンの値を加算
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
					{
						// ※DropOutの有無で違うのはこの一行だけ
						tmp += i_lppInputBuffer[batchNum][inputNum] * this->lppNeuron[neuronNum][inputNum] * this->lppDropOutBuffer[neuronNum][inputNum];
						
					#ifdef _DEBUG
					if(isnan(tmp))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
					}
					tmp += this->lpBias[neuronNum];

					// 活性化関数
					if(this->layerStructure.ActivationType == Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU)
					{
						// ReLU
						this->lpOutputBuffer[batchNum][neuronNum] = max(0.0f, tmp);
					}
					else
					{
						// シグモイド関数を演算
						this->lpOutputBuffer[batchNum][neuronNum] = 1 / (1+exp(-tmp));
					}

					#ifdef _DEBUG
					if(isnan(this->lpOutputBuffer[batchNum][neuronNum]))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
	{
		return &this->lppBatchOutputBuffer[0];
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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

public:
	//================================
	// 学習処理
	//================================
	/** 学習誤差を計算する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		this->m_lppDOutputBuffer = i_lppDOutputBuffer;

		// ループ内でif分を実行すると処理不可がかさむので外で分ける.
		if(!this->onUseDropOut)
		{
			// DropOut無し
			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				// 入力誤差差分を計算
				for(unsigned int inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					float tmp = 0.0f;

					for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						tmp += i_lppDOutputBuffer[batchNum][neuronNum] * this->lppNeuron[neuronNum][inputNum];
					}
#ifdef _DEBUG
					if(isnan(tmp))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

					// 活性化関数で分岐
					if(this->layerStructure.ActivationType == Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU)
					{
						// ReLU
						this->lpDInputBuffer[batchNum][inputNum] = (float)(this->m_lppInputBuffer[batchNum][inputNum] > 0.0f) * tmp;
					}
					else
					{
						// シグモイド
						this->lpDInputBuffer[batchNum][inputNum] = min(1.0f, this->m_lppInputBuffer[batchNum][inputNum]) * (1.0f - min(1.0f, this->m_lppInputBuffer[batchNum][inputNum])) * tmp;
					}
					#ifdef _DEBUG
						if(isnan(lpDInputBuffer[batchNum][inputNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
				}
			}
		}
		else
		{
			// DropOut有り
			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				// 入力誤差差分を計算
				for(unsigned int inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					float tmp = 0.0f;

					for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						// DropOutの有無で違うのはこの一文だけ
						tmp += i_lppDOutputBuffer[batchNum][neuronNum] * this->lppNeuron[neuronNum][inputNum] * this->lppDropOutBuffer[neuronNum][inputNum];

						#ifdef _DEBUG
						if(isnan(i_lppDOutputBuffer[batchNum][neuronNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
						if(isnan(this->lppNeuron[neuronNum][inputNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
						#endif
					}

					#ifdef _DEBUG
					if(isnan(tmp))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif

					// 活性化関数で分岐
					if(this->layerStructure.ActivationType == Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU)
					{
						// ReLU
						this->lpDInputBuffer[batchNum][inputNum] = (float)(1.0f * (this->m_lppInputBuffer[batchNum][inputNum] > 0.0f)) * tmp;
						#ifdef _DEBUG
						if(isnan(this->lpDInputBuffer[batchNum][inputNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
						#endif
					}
					else
					{
						// シグモイド
						this->lpDInputBuffer[batchNum][inputNum] = min(1.0f, this->m_lppInputBuffer[batchNum][inputNum]) * (1.0f - min(1.0f, this->m_lppInputBuffer[batchNum][inputNum])) * tmp;
					}
					#ifdef _DEBUG
					if(isnan(lpDInputBuffer[batchNum][inputNum]))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習差分をレイヤーに反映させる.
		入力信号、出力信号は直前のCalculateの値を参照する.
		出力誤差差分、入力誤差差分は直前のCalculateLearnErrorの値を参照する. */
	ErrorCode ReflectionLearnError(void)
	{
		for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// バイアス更新
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					 sumDOutput += this->m_lppDOutputBuffer[batchNum][neuronNum];
				}
				
				#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
				#endif

				this->lpBias[neuronNum] += this->learnData.LearnCoeff * sumDOutput;// / this->batchSize;
			}

			// 入力対応ニューロン更新
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					sumDOutput += this->m_lppDOutputBuffer[batchNum][neuronNum] * this->m_lppInputBuffer[batchNum][inputNum];
				}
								
				#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
				#endif

				this->lppNeuron[neuronNum][inputNum] += this->learnData.LearnCoeff * sumDOutput;// / this->batchSize;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const
	{
		return &this->lppBatchDInputBuffer[0];
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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

};


/** CPU処理用のレイヤーを作成 */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerCPU(Gravisbell::GUID guid)
{
	return new FeedforwardCPU(guid);
}