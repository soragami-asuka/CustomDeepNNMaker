//======================================
// レイヤー間の接続設定用クラス.
// 出力信号の代用
//======================================
#include"stdafx.h"

#include"LayerConnect.h"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** コンストラクタ */
	LayerConnectSingle2Single::LayerConnectSingle2Single(INNLayer* pLayer)
		:	pLayer	(pLayer)
	{
	}
	/** デストラクタ */
	LayerConnectSingle2Single::~LayerConnectSingle2Single()
	{
		if(pLayer != NULL)
			delete pLayer;
	}

	/** GUIDを取得する */
	Gravisbell::GUID LayerConnectSingle2Single::GetGUID()const
	{
		return this->pLayer->GetGUID();
	}
	/** レイヤー種別の取得.
		ELayerKind の組み合わせ. */
	U32 LayerConnectSingle2Single::GetLayerKind()const
	{
		return this->pLayer->GetLayerKind();
	}

	/** 出力データバッファを取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER LayerConnectSingle2Single::GetOutputBuffer()const
	{
		return this->pLayer->GetOutputBuffer();
	}

	/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
	S32 LayerConnectSingle2Single::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		for(U32 pos=0; pos<this->lppOutputToLayer.size(); pos++)
		{
			if(this->lppOutputToLayer[pos].pLayer->GetGUID() == i_guid)
				return pos;
		}

		return -1;
	}
	/** 入力誤差バッファを位置指定で取得する */
	CONST_BATCH_BUFFER_POINTER LayerConnectSingle2Single::GetDInputBufferByNum(S32 num)const
	{
		return this->pLayer->GetDInputBuffer();
	}

	/** レイヤーリストを作成する.
		@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
	ErrorCode LayerConnectSingle2Single::CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const
	{
		io_lpLayerGUID.insert(this->GetGUID());
		for(auto pInputFromLayer : this->lppInputFromLayer)
			pInputFromLayer->CreateLayerList(io_lpLayerGUID);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 計算順序リストを作成する.
		@param	i_lpLayerGUID		全レイヤーのGUID.
		@param	io_lpCalculateList	演算順に並べられた接続リスト.
		@param	io_lpAddedList		接続リストに登録済みのレイヤーのGUID一覧.
		@param	io_lpAddWaitList	追加待機状態の接続クラスのリスト. */
	ErrorCode LayerConnectSingle2Single::CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)
	{
		// 追加処理
		{
			// 先頭に追加
			io_lpCalculateList.insert(io_lpCalculateList.begin(), this);

			// 追加済に設定
			io_lpAddedList.insert(this->GetGUID());

			// 追加待機状態の場合解除 ※追加待機済みになることはあり得ないのでif文を通ることはないが念のため
			if(io_lpAddWaitList.count(this) > 0)
				io_lpAddWaitList.erase(this);
		}

		// 入力元レイヤーを追加
		for(auto pInputFromLayer : this->lppInputFromLayer)
		{
			if(pInputFromLayer->GetLayerKind() & Gravisbell::Layer::LAYER_KIND_MULT_OUTPUT)
			{
				// 複数出力レイヤーの場合は一旦保留
				io_lpAddWaitList.insert(pInputFromLayer);
			}
			else
			{
				// 単独出力レイヤーの場合は処理を実行
				ErrorCode errCode = pInputFromLayer->CreateCalculateList(i_lpLayerGUID, io_lpCalculateList, io_lpAddedList, io_lpAddWaitList);
				if(errCode != ErrorCode::ERROR_CODE_NONE)
					return errCode;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** レイヤーに入力レイヤーを追加する. */
	ErrorCode LayerConnectSingle2Single::AddInputLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		if(!this->lppInputFromLayer.empty())
			return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;

		// 入力元レイヤーに対して自分を出力先として設定
		ErrorCode err = pInputFromLayer->AddOutputToLayer(this);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 入力元レイヤーのリストに追加
		this->lppInputFromLayer.push_back(pInputFromLayer);

		return ErrorCode::ERROR_CODE_NONE;;
	}
	/** レイヤーにバイパスレイヤーを追加する.*/
	ErrorCode LayerConnectSingle2Single::AddBypassLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}

	/** レイヤーの入力レイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode LayerConnectSingle2Single::ResetInputLayer()
	{
		auto it = this->lppInputFromLayer.begin();
		while(it != this->lppInputFromLayer.end())
		{
			// 入力元から出力先を削除
			(*it)->EraseOutputToLayer(this->GetGUID());

			// 入力元レイヤーを削除
			it = this->lppInputFromLayer.erase(it);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーのバイパスレイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode LayerConnectSingle2Single::ResetBypassLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーに接続している入力レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 LayerConnectSingle2Single::GetInputLayerCount()const
	{
		return this->lppInputFromLayer.size();
	}
	/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectSingle2Single::GetInputLayerByNum(U32 i_inputNum)
	{
		if(i_inputNum >= this->lppInputFromLayer.size())
			return NULL;

		return this->lppInputFromLayer[i_inputNum];
	}

	/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
	U32 LayerConnectSingle2Single::GetBypassLayerCount()const
	{
		return 0;
	}
	/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectSingle2Single::GetBypassLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}


	/** 出力先レイヤーを追加する */
	ErrorCode LayerConnectSingle2Single::AddOutputToLayer(ILayerConnect* pOutputToLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}
	/** 出力先レイヤーを削除する */
	ErrorCode LayerConnectSingle2Single::EraseOutputToLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}


	//=======================================
	// 演算関連
	//=======================================

	/** 接続の確立を行う */
	ErrorCode LayerConnectSingle2Single::EstablishmentConnection(void)
	{
		// 入力元レイヤー数の確認
		if(this->lppInputFromLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppInputFromLayer.size() > 1)
			return ERROR_CODE_FRAUD_INPUT_COUNT;

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
	ErrorCode LayerConnectSingle2Single::PreProcessLearn(unsigned int batchSize)
	{
		return this->pLayer->PreProcessLearn(batchSize);
	}
	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectSingle2Single::PreProcessCalculate(unsigned int batchSize)
	{
		return this->pLayer->PreProcessCalculate(batchSize);
	}

	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectSingle2Single::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		return this->pLayer->PreProcessLearnLoop(data);
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectSingle2Single::PreProcessCalculateLoop()
	{
		return this->PreProcessCalculateLoop();
	}

	/** 演算処理を実行する. */
	ErrorCode LayerConnectSingle2Single::Calculate(void)
	{
		return this->pLayer->Calculate(lppInputFromLayer[0]->GetOutputBuffer());
	}
	/** 学習誤差を計算する. */
	ErrorCode LayerConnectSingle2Single::CalculateLearnError(void)
	{
		return this->pLayer->CalculateLearnError(this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum(this->lppOutputToLayer[0].position));
	}
	/** 学習差分をレイヤーに反映させる.*/
	ErrorCode LayerConnectSingle2Single::ReflectionLearnError(void)
	{
		return this->pLayer->ReflectionLearnError();
	}
	
}	// Gravisbell
}	// Layer
}	// NeuralNetwork