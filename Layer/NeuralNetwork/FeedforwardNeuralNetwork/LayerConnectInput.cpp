//======================================
// レイヤー間の接続設定用クラス.
// 入力信号の代用
//======================================
#include"stdafx.h"

#include"LayerConnectInput.h"
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


	//====================================
	// 実行時設定
	//====================================
	/** 実行時設定を取得する. */
	const SettingData::Standard::IData* LayerConnectInput::GetRuntimeParameter()const
	{
		return NULL;
	}

	/** 実行時設定を設定する.
		int型、float型、enum型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectInput::SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** 実行時設定を設定する.
		int型、float型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectInput::SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** 実行時設定を設定する.
		bool型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectInput::SetRuntimeParameter(const wchar_t* i_dataID, bool i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** 実行時設定を設定する.
		string型が対象.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode LayerConnectInput::SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}

	/** レイヤーの学習禁止設定を取得する */
	bool LayerConnectInput::GetLayerFixFlag()
	{
		return false;
	}

	/** レイヤーに学習禁止を設定する.
		@param	guid		設定対象レイヤーのGUID.
		@param	i_fixFlag	固定化フラグ.true=学習しない. */
	ErrorCode LayerConnectInput::SetLayerFixFlag(bool i_fixFlag)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

		
	//====================================
	// 入出力データ構造
	//====================================

	/** 出力データ構造を取得する.
		@return	出力データ構造 */
	IODataStruct LayerConnectInput::GetOutputDataStruct()const
	{
		return this->neuralNetwork.GetInputDataStruct();
	}
	/** 出力データバッファを取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER LayerConnectInput::GetOutputBuffer_d()const
	{
		return this->neuralNetwork.GetInputBuffer_d();
	}

	/** 入力誤差バッファの位置を入力元レイヤーのGUID指定で取得する */
	S32 LayerConnectInput::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		return 0;
	}
	/** 入力誤差バッファを位置指定で取得する */
	CONST_BATCH_BUFFER_POINTER LayerConnectInput::GetDInputBufferByNum_d(S32 num)const
	{
		if(this->lppOutputToLayer.empty())
			return NULL;
		return this->neuralNetwork.GetDInputBuffer_d();
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

	/** レイヤーから入力レイヤーを削除する */
	ErrorCode LayerConnectInput::EraseInputLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}
	/** レイヤーからバイパスレイヤーを削除する */
	ErrorCode LayerConnectInput::EraseBypassLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
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
	/** レイヤーに接続している入力レイヤーを番号指定で取得する.
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
	/** レイヤーに接続しているバイパスレイヤーを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectInput::GetBypassLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}


	//==========================================
	// 出力レイヤー関連
	//==========================================

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
	
	/** レイヤーに接続している出力先レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 LayerConnectInput::GetOutputToLayerCount()const
	{
		return (U32)this->lppOutputToLayer.size();
	}
	/** レイヤーに接続している出力先レイヤーを番号指定で取得する.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ILayerConnect* LayerConnectInput::GetOutputToLayerByNum(U32 i_num)
	{
		if(i_num >= this->lppOutputToLayer.size())
			return NULL;

		return this->lppOutputToLayer[i_num].pLayer;
	}

	
	/** レイヤーの接続を解除 */
	ErrorCode LayerConnectInput::Disconnect(void)
	{
		// 出力先レイヤーから自分を削除
		for(auto it : this->lppOutputToLayer)
			it.pLayer->EraseInputLayer(this->GetGUID());

		// 出力先レイヤーを全削除
		this->lppOutputToLayer.clear();

		this->ResetInputLayer();
		this->ResetBypassLayer();

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** レイヤーで使用する出力バッファのIDを登録する */
	ErrorCode LayerConnectInput::SetOutputBufferID(S32 i_outputBufferID)
	{
		// 入力レイヤーの出力バッファはニューラルネットワークの入力バッファと同一なので、出力バッファを割り当てる必要はない.
		return ErrorCode::ERROR_CODE_NONE;
	}
	
	/** レイヤーで使用する入力誤差バッファのIDを取得する
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	ErrorCode LayerConnectInput::SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーで使用する入力誤差バッファのIDを取得する
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定. */
	S32 LayerConnectInput::GetDInputBufferID(U32 i_inputNum)const
	{
		return -1;
	}


	//=======================================
	// 演算関連
	//=======================================
	
	/** レイヤーの初期化処理.
		接続状況は維持したままレイヤーの中身を初期化する. */
	ErrorCode LayerConnectInput::Initialize(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

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
	
	/** 処理ループの初期化処理.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LayerConnectInput::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する. */
	ErrorCode LayerConnectInput::Calculate(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 入力誤差計算を実行する. */
	ErrorCode LayerConnectInput::CalculateDInput(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 学習処理を実行する. */
	ErrorCode LayerConnectInput::Training(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

}	// Gravisbell
}	// Layer
}	// NeuralNetwork