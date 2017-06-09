//======================================
// レイヤー間の接続設定用クラス.
// 出力信号の代用
//======================================
#include"stdafx.h"

#include"LayerConnectSingle2Single.h"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** コンストラクタ */
	LayerConnectSingle2Single::LayerConnectSingle2Single(class FeedforwardNeuralNetwork_Base& neuralNetwork, ILayerBase* pLayer, Gravisbell::SettingData::Standard::IData* pLearnSettingData)
		:	neuralNetwork		(neuralNetwork)
		,	pLayer				(pLayer)
		,	pLayer_io			(dynamic_cast<INNSingle2SingleLayer*>(pLayer))
		,	pLearnSettingData	(pLearnSettingData)
		,	dInputBufferID		(INVALID_DINPUTBUFFER_ID)
	{
	}
	/** デストラクタ */
	LayerConnectSingle2Single::~LayerConnectSingle2Single()
	{
		if(pLayer != NULL)
			delete pLayer;
		if(pLearnSettingData != NULL)
			delete pLearnSettingData;
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
	
	/** 学習設定のポインタを取得する.
		取得したデータを直接書き換えることで次の学習ループに反映されるが、NULLが返ってくることもあるので注意. */
	Gravisbell::SettingData::Standard::IData* LayerConnectSingle2Single::GetLearnSettingData()
	{
		return this->pLearnSettingData;
	}

	/** 出力データ構造を取得する.
		@return	出力データ構造 */
	IODataStruct LayerConnectSingle2Single::GetOutputDataStruct()const
	{
		return this->pLayer_io->GetOutputDataStruct();
	}
	/** 出力データバッファを取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER LayerConnectSingle2Single::GetOutputBuffer()const
	{
		return this->pLayer_io->GetOutputBuffer();
	}

	/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
	S32 LayerConnectSingle2Single::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		for(U32 pos=0; pos<this->lppInputFromLayer.size(); pos++)
		{
			if(this->lppInputFromLayer[pos]->GetGUID() == i_guid)
				return pos;
		}

		return -1;
	}
	/** 入力誤差バッファを位置指定で取得する */
	CONST_BATCH_BUFFER_POINTER LayerConnectSingle2Single::GetDInputBufferByNum(S32 num)const
	{
		return neuralNetwork.GetDInputBuffer(this->GetDInputBufferID(0));
	}

	/** レイヤーリストを作成する.
		@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
	ErrorCode LayerConnectSingle2Single::CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const
	{
		if(io_lpLayerGUID.count(this->GetGUID()))
			return ErrorCode::ERROR_CODE_NONE;

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


	/** レイヤーから入力レイヤーを削除する */
	ErrorCode LayerConnectSingle2Single::EraseInputLayer(const Gravisbell::GUID& guid)
	{
		auto it = this->lppInputFromLayer.begin();
		while(it != this->lppInputFromLayer.end())
		{
			if((*it)->GetGUID() == guid)
			{
				it = this->lppInputFromLayer.erase(it);
				return ErrorCode::ERROR_CODE_NONE;
			}
			else
			{
				it++;
			}
		}

		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}
	/** レイヤーからバイパスレイヤーを削除する */
	ErrorCode LayerConnectSingle2Single::EraseBypassLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
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
		return (U32)this->lppInputFromLayer.size();
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

	
	//==========================================
	// 出力レイヤー関連
	//==========================================

	/** 出力先レイヤーを追加する */
	ErrorCode LayerConnectSingle2Single::AddOutputToLayer(ILayerConnect* pOutputToLayer)
	{
		if(!this->lppOutputToLayer.empty())
			return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
		this->lppOutputToLayer.push_back(pOutputToLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 出力先レイヤーを削除する */
	ErrorCode LayerConnectSingle2Single::EraseOutputToLayer(const Gravisbell::GUID& guid)
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

	
	/** レイヤーに接続している出力先レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 LayerConnectSingle2Single::GetOutputToLayerCount()const
	{
		return (U32)this->lppOutputToLayer.size();
	}
	/** レイヤーに接続している出力先レイヤーを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectSingle2Single::GetOutputToLayerByNum(U32 i_num)
	{
		if(i_num >= this->lppOutputToLayer.size())
			return NULL;

		return this->lppOutputToLayer[i_num].pLayer;
	}



	//=======================================
	// 接続関連
	//=======================================

	/** レイヤーの接続を解除 */
	ErrorCode LayerConnectSingle2Single::Disconnect(void)
	{
		// 出力先レイヤーから自分を削除
		for(auto it : this->lppOutputToLayer)
			it.pLayer->EraseInputLayer(this->GetGUID());

		// 出力先レイヤーを全削除
		this->lppOutputToLayer.clear();

		// 自身の入力レイヤー/バイパスレイヤーを削除
		this->ResetInputLayer();
		this->ResetBypassLayer();

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** レイヤーで使用する入力誤差バッファのIDを取得する
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	S32 LayerConnectSingle2Single::GetDInputBufferID(U32 i_inputNum)const
	{
		return this->dInputBufferID;
	}
	/** レイヤーで使用する入力誤差バッファのIDを取得する
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ErrorCode LayerConnectSingle2Single::SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID)
	{
		this->dInputBufferID = i_DInputBufferID;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//=======================================
	// 演算関連
	//=======================================

	/** レイヤーの初期化処理.
		接続状況は維持したままレイヤーの中身を初期化する. */
	ErrorCode LayerConnectSingle2Single::Initialize(void)
	{
		return this->pLayer->Initialize();
	}

	/** 接続の確立を行う */
	ErrorCode LayerConnectSingle2Single::EstablishmentConnection(void)
	{
		// 入力元レイヤー数の確認
		if(this->lppInputFromLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppInputFromLayer.size() > 1)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 入力元レイヤーのバッファ数を確認
		if(this->lppInputFromLayer[0]->GetOutputDataStruct().GetDataCount() != this->pLayer_io->GetInputBufferCount())
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力先レイヤー数の確認
		if(this->lppOutputToLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppOutputToLayer.size() > 1)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力先レイヤーの位置を登録
		this->lppOutputToLayer[0].position = this->lppOutputToLayer[0].pLayer->GetDInputPositionByGUID(this->GetGUID());

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
	ErrorCode LayerConnectSingle2Single::PreProcessLearnLoop()
	{
		if(this->pLearnSettingData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		return this->pLayer->PreProcessLearnLoop(*this->pLearnSettingData);
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectSingle2Single::PreProcessCalculateLoop()
	{
		return this->pLayer->PreProcessCalculateLoop();
	}

	/** 演算処理を実行する. */
	ErrorCode LayerConnectSingle2Single::Calculate(void)
	{
		return this->pLayer_io->Calculate(lppInputFromLayer[0]->GetOutputBuffer());
	}
	/** 学習誤差を計算する. */
	ErrorCode LayerConnectSingle2Single::Training(void)
	{
		if(this->GetDInputBufferID(0) < 0)
		{
			return this->pLayer_io->Training(
				this->neuralNetwork.GetDInputBuffer(),
				this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum(this->lppOutputToLayer[0].position) );
		}
		else
		{
			return this->pLayer_io->Training(
				this->neuralNetwork.GetDInputBuffer(this->GetDInputBufferID(0)),
				this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum(this->lppOutputToLayer[0].position) );
		}
	}


}	// Gravisbell
}	// Layer
}	// NeuralNetwork