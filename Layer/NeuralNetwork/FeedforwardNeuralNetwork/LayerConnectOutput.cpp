//======================================
// レイヤー間の接続設定用クラス.
// 出力信号の代用
//======================================
#include"stdafx.h"

#include"LayerConnectOutput.h"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** コンストラクタ */
	LayerConnectOutput::LayerConnectOutput(class FeedforwardNeuralNetwork_Base& neuralNetwork)
		:	neuralNetwork	(neuralNetwork)
		,	dInputBufferID	(INVALID_DINPUTBUFFER_ID)	/**< 入力誤差バッファID */
	{
	}
	/** デストラクタ */
	LayerConnectOutput::~LayerConnectOutput()
	{
	}

	/** GUIDを取得する */
	Gravisbell::GUID LayerConnectOutput::GetGUID()const
	{
		return this->neuralNetwork.GetGUID();
	}
	/** レイヤー種別の取得.
		ELayerKind の組み合わせ. */
	U32 LayerConnectOutput::GetLayerKind()const
	{
		return (this->neuralNetwork.GetLayerKind() & Gravisbell::Layer::LAYER_KIND_CALCTYPE) | Gravisbell::Layer::LAYER_KIND_SINGLE_INPUT;
	}


	//====================================
	// 実行時設定
	//====================================
	/** 実行時設定を取得する. */
	const SettingData::Standard::IData* LayerConnectOutput::GetRuntimeParameter()const
	{
		return NULL;
	}

	/** 実行時設定を設定する.
		int型、float型、enum型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectOutput::SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** 実行時設定を設定する.
		int型、float型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectOutput::SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** 実行時設定を設定する.
		bool型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectOutput::SetRuntimeParameter(const wchar_t* i_dataID, bool i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** 実行時設定を設定する.
		string型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectOutput::SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}


		
	//====================================
	// 入出力データ構造
	//====================================
	/** 出力データ構造を取得する.
		@return	出力データ構造 */
	IODataStruct LayerConnectOutput::GetOutputDataStruct()const
	{
		if(this->lppInputFromLayer.empty())
			return IODataStruct();

		return this->lppInputFromLayer[0]->GetOutputDataStruct();
	}
	/** 出力データバッファを取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER LayerConnectOutput::GetOutputBuffer_d()const
	{
		if(this->lppInputFromLayer.empty())
			return NULL;

		return this->lppInputFromLayer[0]->GetOutputBuffer_d();
	}

	/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
	S32 LayerConnectOutput::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		return 0;
	}
	/** 入力誤差バッファを位置指定で取得する */
	CONST_BATCH_BUFFER_POINTER LayerConnectOutput::GetDInputBufferByNum_d(S32 num)const
	{
		return this->neuralNetwork.GetDOutputBuffer_d();
	}

	/** レイヤーリストを作成する.
		@param	i_lpLayerGUID	接続しているGUIDのリスト.入力方向に確認する. */
	ErrorCode LayerConnectOutput::CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const
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
	ErrorCode LayerConnectOutput::CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)
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
	ErrorCode LayerConnectOutput::AddInputLayerToLayer(ILayerConnect* pInputFromLayer)
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
	ErrorCode LayerConnectOutput::AddBypassLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}


	/** レイヤーから入力レイヤーを削除する */
	ErrorCode LayerConnectOutput::EraseInputLayer(const Gravisbell::GUID& guid)
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
	ErrorCode LayerConnectOutput::EraseBypassLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}


	/** レイヤーの入力レイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode LayerConnectOutput::ResetInputLayer()
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
	ErrorCode LayerConnectOutput::ResetBypassLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーに接続している入力レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 LayerConnectOutput::GetInputLayerCount()const
	{
		return (U32)this->lppInputFromLayer.size();
	}
	/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectOutput::GetInputLayerByNum(U32 i_inputNum)
	{
		if(i_inputNum >= this->lppInputFromLayer.size())
			return NULL;

		return this->lppInputFromLayer[i_inputNum];
	}

	/** レイヤーに接続しているバイパスレイヤーの数を取得する. */
	U32 LayerConnectOutput::GetBypassLayerCount()const
	{
		return 0;
	}
	/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectOutput::GetBypassLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}
	

	//==========================================
	// 出力レイヤー関連
	//==========================================

	/** 出力先レイヤーを追加する */
	ErrorCode LayerConnectOutput::AddOutputToLayer(ILayerConnect* pOutputToLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}
	/** 出力先レイヤーを削除する */
	ErrorCode LayerConnectOutput::EraseOutputToLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}
	
	/** レイヤーに接続している出力先レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 LayerConnectOutput::GetOutputToLayerCount()const
	{
		return 0;
	}
	/** レイヤーに接続している出力先レイヤーを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectOutput::GetOutputToLayerByNum(U32 i_num)
	{
		return NULL;
	}



	//=======================================
	// 接続関連
	//=======================================
	/** レイヤーの接続を解除 */
	ErrorCode LayerConnectOutput::Disconnect(void)
	{
		this->ResetInputLayer();
		this->ResetBypassLayer();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーで使用する出力バッファのIDを登録する */
	ErrorCode LayerConnectOutput::SetOutputBufferID(S32 i_outputBufferID)
	{
		// 出力レイヤーの出力バッファは直前のレイヤーの出力バッファと同一のため、登録不要

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーで使用する入力誤差バッファのIDを取得する
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	S32 LayerConnectOutput::GetDInputBufferID(U32 i_inputNum)const
	{
		return this->dInputBufferID;
	}
	/** レイヤーで使用する入力誤差バッファのIDを取得する
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ErrorCode LayerConnectOutput::SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID)
	{
		this->dInputBufferID = i_DInputBufferID;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//=======================================
	// 演算関連
	//=======================================
	/** レイヤーの初期化処理.
		接続状況は維持したままレイヤーの中身を初期化する. */
	ErrorCode LayerConnectOutput::Initialize(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 接続の確立を行う */
	ErrorCode LayerConnectOutput::EstablishmentConnection(void)
	{
		// 入力元レイヤー数の確認
		if(this->lppInputFromLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppInputFromLayer.size() > 1)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode LayerConnectOutput::PreProcessLearn(unsigned int batchSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectOutput::PreProcessCalculate(unsigned int batchSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	
	/** 処理ループの初期化処理.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectOutput::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する. */
	ErrorCode LayerConnectOutput::Calculate(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 学習処理を実行する. */
	ErrorCode LayerConnectOutput::CalculateDInput(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 学習処理を実行する. */
	ErrorCode LayerConnectOutput::Training(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

}	// Gravisbell
}	// Layer
}	// NeuralNetwork