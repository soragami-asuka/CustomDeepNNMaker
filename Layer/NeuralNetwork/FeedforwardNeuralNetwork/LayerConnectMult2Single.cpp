//======================================
// レイヤー間の接続設定用クラス.
// 出力信号の代用
//======================================
#include"stdafx.h"

#include"LayerConnectMult2Single.h"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** コンストラクタ */
	LayerConnectMult2Single::LayerConnectMult2Single(class FeedforwardNeuralNetwork_Base& neuralNetwork, ILayerBase* pLayer, bool onFixFlag)
		:	neuralNetwork		(neuralNetwork)
		,	pLayer				(pLayer)
		,	pLayer_io			(dynamic_cast<INNMult2SingleLayer*>(pLayer))
		,	outputBufferID		(INVALID_OUTPUTBUFFER_ID)	/**< 出力バッファID */
		,	onLayerFix			(onFixFlag)					/**< レイヤー固定化フラグ */
		,	isNecessaryBackPropagation	(true)	/**< 誤差伝搬が必要なフラグ. falseの場合、ニューラルネットワーク自体が入力誤差バッファを持たない場合は誤差伝搬しない */
	{
	}
	/** デストラクタ */
	LayerConnectMult2Single::~LayerConnectMult2Single()
	{
		if(pLayer != NULL)
			delete pLayer;
	}

	/** GUIDを取得する */
	Gravisbell::GUID LayerConnectMult2Single::GetGUID()const
	{
		return this->pLayer->GetGUID();
	}
	/** レイヤー種別の取得.
		ELayerKind の組み合わせ. */
	U32 LayerConnectMult2Single::GetLayerKind()const
	{
		return this->pLayer->GetLayerKind();
	}


	//====================================
	// 実行時設定
	//====================================
	/** 実行時設定を取得する. */
	const SettingData::Standard::IData* LayerConnectMult2Single::GetRuntimeParameter()const
	{
		return this->pLayer->GetRuntimeParameter();
	}

	/** 実行時設定を設定する.
		int型、float型、enum型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectMult2Single::SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param)
	{
		return this->pLayer->SetRuntimeParameter(i_dataID, i_param);
	}
	/** 実行時設定を設定する.
		int型、float型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectMult2Single::SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param)
	{
		return this->pLayer->SetRuntimeParameter(i_dataID, i_param);
	}
	/** 実行時設定を設定する.
		bool型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectMult2Single::SetRuntimeParameter(const wchar_t* i_dataID, bool i_param)
	{
		return this->pLayer->SetRuntimeParameter(i_dataID, i_param);
	}
	/** 実行時設定を設定する.
		string型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectMult2Single::SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param)
	{
		return this->pLayer->SetRuntimeParameter(i_dataID, i_param);
	}

		
	//====================================
	// 入出力データ構造
	//====================================

	/** 出力データ構造を取得する.
		@return	出力データ構造 */
	IODataStruct LayerConnectMult2Single::GetOutputDataStruct()const
	{
		return this->pLayer_io->GetOutputDataStruct();
	}
	/** 出力データバッファを取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER LayerConnectMult2Single::GetOutputBuffer_d()const
	{
		return this->neuralNetwork.ReserveOutputBuffer_d(this->outputBufferID, this->GetGUID());
	}

	/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
	S32 LayerConnectMult2Single::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		for(U32 pos=0; pos<this->lppInputFromLayer.size(); pos++)
		{
			if(this->lppInputFromLayer[pos]->GetGUID() == i_guid)
				return pos;
		}

		return -1;
	}
	/** 入力誤差バッファを位置指定で取得する */
	CONST_BATCH_BUFFER_POINTER LayerConnectMult2Single::GetDInputBufferByNum_d(S32 num)const
	{
		return this->neuralNetwork.GetTmpDInputBuffer_d(this->GetDInputBufferID(num));
	}

	/** レイヤーリストを作成する.
		@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
	ErrorCode LayerConnectMult2Single::CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const
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
	ErrorCode LayerConnectMult2Single::CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)
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
	ErrorCode LayerConnectMult2Single::AddInputLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		// 入力元レイヤーに対して自分を出力先として設定
		ErrorCode err = pInputFromLayer->AddOutputToLayer(this);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 入力元レイヤーのリストに追加
		this->lppInputFromLayer.push_back(pInputFromLayer);

		return ErrorCode::ERROR_CODE_NONE;;
	}
	/** レイヤーにバイパスレイヤーを追加する.*/
	ErrorCode LayerConnectMult2Single::AddBypassLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}


	/** レイヤーから入力レイヤーを削除する */
	ErrorCode LayerConnectMult2Single::EraseInputLayer(const Gravisbell::GUID& guid)
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
	ErrorCode LayerConnectMult2Single::EraseBypassLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}


	/** レイヤーの入力レイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode LayerConnectMult2Single::ResetInputLayer()
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
	ErrorCode LayerConnectMult2Single::ResetBypassLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーに接続している入力レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 LayerConnectMult2Single::GetInputLayerCount()const
	{
		return (U32)this->lppInputFromLayer.size();
	}
	/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectMult2Single::GetInputLayerByNum(U32 i_inputNum)
	{
		if(i_inputNum >= this->lppInputFromLayer.size())
			return NULL;

		return this->lppInputFromLayer[i_inputNum];
	}

	/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
	U32 LayerConnectMult2Single::GetBypassLayerCount()const
	{
		return 0;
	}
	/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectMult2Single::GetBypassLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}


	//==========================================
	// 学習フラグ関連
	//==========================================
	/** 学習固定レイヤーフラグ.
		学習固定レイヤー(学習が必要ないレイヤー)の場合trueが返る. */
	bool LayerConnectMult2Single::IsFixLayer(void)const
	{
		return this->onLayerFix;
	}

	/** 入力誤差の計算が必要なフラグ.
		必要な場合trueが返る. */
	bool LayerConnectMult2Single::IsNecessaryCalculateDInput(void)const
	{
		if(this->lppInputFromLayer.empty())
			return false;

		// 一つ前のレイヤーが誤差伝搬を必要とする場合は入力誤差計算を実行する
		for(U32 layerNum=0; layerNum<this->lppInputFromLayer.size(); layerNum++)
		{
			if(this->lppInputFromLayer[0]->IsNecessaryBackPropagation())
				return true;
		}
		return false;
	}

	/** 誤差伝搬が必要なフラグ.
		誤差伝搬が必要な場合はtrueが返る.falseが返った場合、これ以降誤差伝搬を一切必要としない. */
	bool LayerConnectMult2Single::IsNecessaryBackPropagation(void)const
	{
		if(this->isNecessaryBackPropagation)
			return true;

		// ニューラルネットワーク本体の入力誤差信号が存在するか
		if(this->neuralNetwork.CheckIsHaveDInputBuffer())
			return true;

		return false;
	}
	
	//==========================================
	// 出力レイヤー関連
	//==========================================

	/** 出力先レイヤーを追加する */
	ErrorCode LayerConnectMult2Single::AddOutputToLayer(ILayerConnect* pOutputToLayer)
	{
		if(!this->lppOutputToLayer.empty())
			return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
		this->lppOutputToLayer.push_back(pOutputToLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 出力先レイヤーを削除する */
	ErrorCode LayerConnectMult2Single::EraseOutputToLayer(const Gravisbell::GUID& guid)
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
	U32 LayerConnectMult2Single::GetOutputToLayerCount()const
	{
		return (U32)this->lppOutputToLayer.size();
	}
	/** レイヤーに接続している出力先レイヤーを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectMult2Single::GetOutputToLayerByNum(U32 i_num)
	{
		if(i_num >= this->lppOutputToLayer.size())
			return NULL;

		return this->lppOutputToLayer[i_num].pLayer;
	}



	//=======================================
	// 接続関連
	//=======================================

	/** レイヤーの接続を解除 */
	ErrorCode LayerConnectMult2Single::Disconnect(void)
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


	/** レイヤーで使用する出力バッファのIDを登録する */
	ErrorCode LayerConnectMult2Single::SetOutputBufferID(S32 i_outputBufferID)
	{
		this->outputBufferID = i_outputBufferID;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーで使用する入力誤差バッファのIDを取得する
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ErrorCode LayerConnectMult2Single::SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID)
	{
		if(i_inputNum >= this->lpDInputBufferID.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpDInputBufferID[i_inputNum] = i_DInputBufferID;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーで使用する入力誤差バッファのIDを取得する
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	S32 LayerConnectMult2Single::GetDInputBufferID(U32 i_inputNum)const
	{
		if(i_inputNum >= this->lpDInputBufferID.size())
			return INVALID_DINPUTBUFFER_ID;

		return this->lpDInputBufferID[i_inputNum];
	}



	//=======================================
	// 演算関連
	//=======================================

	/** レイヤーの初期化処理.
		接続状況は維持したままレイヤーの中身を初期化する. */
	ErrorCode LayerConnectMult2Single::Initialize(void)
	{
		return this->pLayer->Initialize();
	}

	/** 接続の確立を行う */
	ErrorCode LayerConnectMult2Single::EstablishmentConnection(void)
	{
		// 入力元レイヤー数の確認
		if(this->lppInputFromLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppInputFromLayer.size() != this->pLayer_io->GetInputDataCount())
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 入力元レイヤーのバッファ数を確認
		for(U32 inputNum=0; inputNum<this->lppInputFromLayer.size(); inputNum++)
		{
			if(this->lppInputFromLayer[inputNum]->GetOutputDataStruct().GetDataCount() != this->pLayer_io->GetInputBufferCount(inputNum))
				return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		}

		// 出力先レイヤー数の確認
		if(this->lppOutputToLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppOutputToLayer.size() > 1)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力先レイヤーの位置を登録
		this->lppOutputToLayer[0].position = this->lppOutputToLayer[0].pLayer->GetDInputPositionByGUID(this->GetGUID());

		// 入力誤差バッファID一覧のサイズ更新
		this->lpDInputBufferID.resize(this->lppInputFromLayer.size(), INVALID_DINPUTBUFFER_ID);

		// 誤差伝搬が必要か確認する
		if(!this->onLayerFix)
			this->isNecessaryBackPropagation = true;
		else
		{
			this->isNecessaryBackPropagation = false;
			for(U32 layerNum=0; layerNum<this->lppInputFromLayer.size(); layerNum++)
			{
				if(this->lppInputFromLayer[0]->IsNecessaryBackPropagation())
				{
					this->isNecessaryBackPropagation = true;
					break;
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode LayerConnectMult2Single::PreProcessLearn(unsigned int batchSize)
	{
		// 入力用のバッファを作成
		this->lppInputBuffer.resize(this->lppInputFromLayer.size());
		this->lppDInputBuffer.resize(this->lppInputFromLayer.size());

		return this->pLayer->PreProcessLearn(batchSize);
	}
	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectMult2Single::PreProcessCalculate(unsigned int batchSize)
	{
		// 入力用のバッファを作成
		this->lppInputBuffer.resize(this->lppInputFromLayer.size());

		return this->pLayer->PreProcessCalculate(batchSize);
	}
	
	/** 処理ループの初期化処理.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectMult2Single::PreProcessLoop()
	{
		return this->pLayer->PreProcessLoop();
	}

	/** 演算処理を実行する. */
	ErrorCode LayerConnectMult2Single::Calculate(void)
	{
		// 入力配列を作成
		for(U32 inputNum=0; inputNum<this->lppInputFromLayer.size(); inputNum++)
		{
			this->lppInputBuffer[inputNum] = this->lppInputFromLayer[inputNum]->GetOutputBuffer_d();
		}

		return this->pLayer_io->Calculate_device(
			&this->lppInputBuffer[0],
			neuralNetwork.ReserveOutputBuffer_d(this->outputBufferID, this->GetGUID()) );
	}
	/** 学習誤差を計算する. */
	ErrorCode LayerConnectMult2Single::CalculateDInput(void)
	{
		if(!this->IsNecessaryCalculateDInput())
			return ErrorCode::ERROR_CODE_NONE;

		// 入力/入力誤差配列を作成
		for(U32 inputNum=0; inputNum<this->lppInputFromLayer.size(); inputNum++)
		{
			this->lppInputBuffer[inputNum] = this->lppInputFromLayer[inputNum]->GetOutputBuffer_d();

			if(this->GetDInputBufferID(inputNum) & NETWORK_DINPUTBUFFER_ID_FLAGBIT)
				this->lppDInputBuffer[inputNum] = this->neuralNetwork.GetDInputBuffer_d(this->GetDInputBufferID(inputNum) & 0xFFFF);
			else
				this->lppDInputBuffer[inputNum] = neuralNetwork.GetTmpDInputBuffer_d(this->GetDInputBufferID(inputNum));
		}

		return this->pLayer_io->CalculateDInput_device(
			&this->lppInputBuffer[0],
			&this->lppDInputBuffer[0],
			this->GetOutputBuffer_d(),
			this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum_d(this->lppOutputToLayer[0].position));
	}
	/** 学習誤差を計算する. */
	ErrorCode LayerConnectMult2Single::Training(void)
	{
		if(this->onLayerFix)
			return this->CalculateDInput();

		// 入力/入力誤差配列を作成
		for(U32 inputNum=0; inputNum<this->lppInputFromLayer.size(); inputNum++)
		{
			this->lppInputBuffer[inputNum] = this->lppInputFromLayer[inputNum]->GetOutputBuffer_d();
			
			if(this->GetDInputBufferID(inputNum) & NETWORK_DINPUTBUFFER_ID_FLAGBIT)
				this->lppDInputBuffer[inputNum] = this->neuralNetwork.GetDInputBuffer_d(this->GetDInputBufferID(inputNum) & 0xFFFF);
			else
				this->lppDInputBuffer[inputNum] = neuralNetwork.GetTmpDInputBuffer_d(this->GetDInputBufferID(inputNum));
		}

		return this->pLayer_io->Training_device(
			&this->lppInputBuffer[0],
			&this->lppDInputBuffer[0],
			this->GetOutputBuffer_d(),
			this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum_d(this->lppOutputToLayer[0].position));
	}


}	// Gravisbell
}	// Layer
}	// NeuralNetwork