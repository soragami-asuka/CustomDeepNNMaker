//======================================
// レイヤー間の接続設定用クラス.
// 入力信号の代用
//======================================
#include"stdafx.h"

#include"LayerConnect.h"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** コンストラクタ */
	LayerConnectInput::LayerConnectInput(class FeedforwardNeuralNetwork_Base& neuralNetwork)
		:	neuralNetwork	(neuralNetwork)
	{
	}
	/** デストラクタ */
	LayerConnectInput::~LayerConnectInput()
	{
	}

	/** GUIDを取得する */
	Gravisbell::GUID LayerConnectInput::GetGUID()const
	{
		return this->neuralNetwork.GetInputGUID();
	}
	/** レイヤー種別の取得.
		ELayerKind の組み合わせ. */
	U32 LayerConnectInput::GetLayerKind()const
	{
		return (this->neuralNetwork.GetLayerKind() & Gravisbell::Layer::LAYER_KIND_CALCTYPE) | Gravisbell::Layer::LAYER_KIND_SINGLE_OUTPUT;
	}

	/** 出力データバッファを取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER LayerConnectInput::GetOutputBuffer()const
	{
		return this->neuralNetwork.GetInputBuffer();
	}

	/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
	S32 LayerConnectInput::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		return 0;
	}
	/** 入力誤差バッファを位置指定で取得する */
	CONST_BATCH_BUFFER_POINTER LayerConnectInput::GetDInputBufferByNum(S32 num)const
	{
		if(this->lppOutputToLayer.empty())
			return NULL;
		return (*this->lppOutputToLayer.begin()).pLayer->GetDInputBufferByNum((*this->lppOutputToLayer.begin()).position);
	}

	/** レイヤーリストを作成する.
		@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
	ErrorCode LayerConnectInput::CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const
	{
		io_lpLayerGUID.insert(this->GetGUID());

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 計算順序リストを作成する.
		@param	i_lpLayerGUID		全レイヤーのGUID.
		@param	io_lpCalculateList	演算順に並べられた接続リスト.
		@param	io_lpAddedList		接続リストに登録済みのレイヤーのGUID一覧.
		@param	io_lpAddWaitList	追加待機状態の接続クラスのリスト. */
	ErrorCode LayerConnectInput::CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)
	{
		// 先頭に追加
		io_lpCalculateList.insert(io_lpCalculateList.begin(), this);

		// 追加済に設定
		io_lpAddedList.insert(this->GetGUID());

		// 追加待機状態の場合解除 ※追加待機済みになることはあり得ないのでif文を通ることはない
		if(io_lpAddWaitList.count(this) > 0)
			io_lpAddWaitList.erase(this);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** レイヤーに入力レイヤーを追加する. */
	ErrorCode LayerConnectInput::AddInputLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}
	/** レイヤーにバイパスレイヤーを追加する.*/
	ErrorCode LayerConnectInput::AddBypassLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}

	/** レイヤーの入力レイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode LayerConnectInput::ResetInputLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーのバイパスレイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode LayerConnectInput::ResetBypassLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーに接続している入力レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 LayerConnectInput::GetInputLayerCount()const
	{
		return 0;
	}
	/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectInput::GetInputLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}

	/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
	U32 LayerConnectInput::GetBypassLayerCount()const
	{
		return 0;
	}
	/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectInput::GetBypassLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}


	/** 出力先レイヤーを追加する */
	ErrorCode LayerConnectInput::AddOutputToLayer(ILayerConnect* pOutputToLayer)
	{
		if(!this->lppOutputToLayer.empty())
			return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
		this->lppOutputToLayer.push_back(pOutputToLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 出力先レイヤーを削除する */
	ErrorCode LayerConnectInput::EraseOutputToLayer(const Gravisbell::GUID& guid)
	{
		auto it = this->lppOutputToLayer.begin();
		while(it != this->lppOutputToLayer.end())
		{
			if((*it).pLayer->GetGUID() == guid)
			{
				this->lppOutputToLayer.erase(it);
				return ErrorCode::ERROR_CODE_NONE;
			}
			it++;
		}
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}


	//=======================================
	// 演算関連
	//=======================================

	/** 接続の確立を行う */
	ErrorCode LayerConnectInput::EstablishmentConnection(void)
	{
		// 出力先レイヤー数の確認
		if(this->lppOutputToLayer.empty())
			return ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppOutputToLayer.size() > 1)
			return ERROR_CODE_FRAUD_INPUT_COUNT;


		// 出力先レイヤーの位置を登録
		auto it = this->lppOutputToLayer.begin();
		while(it != this->lppOutputToLayer.end())
		{
			it->position = it->pLayer->GetDInputPositionByGUID(this->GetGUID());

			it++;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode LayerConnectInput::PreProcessLearn(unsigned int batchSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectInput::PreProcessCalculate(unsigned int batchSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectInput::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectInput::PreProcessCalculateLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する. */
	ErrorCode LayerConnectInput::Calculate(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 学習誤差を計算する. */
	ErrorCode LayerConnectInput::CalculateLearnError(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 学習差分をレイヤーに反映させる.*/
	ErrorCode LayerConnectInput::ReflectionLearnError(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

}	// Gravisbell
}	// Layer
}	// NeuralNetwork