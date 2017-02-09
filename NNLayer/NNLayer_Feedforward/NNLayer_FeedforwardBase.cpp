//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward.h"
#include"NNLayer_FeedforwardBase.h"



/** コンストラクタ */
NNLayer_FeedforwardBase::NNLayer_FeedforwardBase(GUID guid)
	:	INNLayer()
	,	guid			(guid)
	,	pConfig			(NULL)
	,	lppInputFromLayer	()		/**< 入力元レイヤーのリスト */
	,	lppOutputToLayer()	/**< 出力先レイヤーのリスト */
{
}

/** デストラクタ */
NNLayer_FeedforwardBase::~NNLayer_FeedforwardBase()
{
	if(pConfig != NULL)
		delete pConfig;
}


//===========================
// レイヤー共通
//===========================
/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
unsigned int NNLayer_FeedforwardBase::GetUseBufferByteCount()const
{
	unsigned int bufferSize = 0;

	if(pConfig == NULL)
		return 0;

	// 設定情報
	bufferSize += pConfig->GetUseBufferByteCount();

	// 本体のバイト数
	bufferSize += (this->GetNeuronCount() * this->GetInputBufferCount()) * sizeof(NEURON_TYPE);	// ニューロン係数
	bufferSize += this->GetNeuronCount() * sizeof(NEURON_TYPE);	// バイアス係数


	return bufferSize;
}

/** レイヤー固有のGUIDを取得する */
ELayerErrorCode NNLayer_FeedforwardBase::GetGUID(GUID& o_guid)const
{
	o_guid = this->guid;

	return LAYER_ERROR_NONE;
}

/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
ELayerErrorCode NNLayer_FeedforwardBase::GetLayerCode(GUID& o_layerCode)const
{
	return ::GetLayerCode(o_layerCode);
}

/** 設定情報を設定 */
ELayerErrorCode NNLayer_FeedforwardBase::SetLayerConfig(const INNLayerConfig& config)
{
	ELayerErrorCode err = LAYER_ERROR_NONE;

	// レイヤーコードを確認
	{
		GUID config_guid;
		err = config.GetLayerCode(config_guid);
		if(err != LAYER_ERROR_NONE)
			return err;

		GUID layer_guid;
		err = ::GetLayerCode(layer_guid);
		if(err != LAYER_ERROR_NONE)
			return err;

		if(config_guid != layer_guid)
			return LAYER_ERROR_INITLAYER_DISAGREE_CONFIG;
	}

	if(this->pConfig != NULL)
		delete this->pConfig;
	this->pConfig = config.Clone();

	return LAYER_ERROR_NONE;
}
/** レイヤーの設定情報を取得する */
const INNLayerConfig* NNLayer_FeedforwardBase::GetLayerConfig()const
{
	return this->pConfig;
}


//===========================
// 入力レイヤー関連
//===========================
/** 入力バッファ数を取得する. */
unsigned int NNLayer_FeedforwardBase::GetInputBufferCount()const
{
	unsigned int intputBufferCount = 0;
	for(auto layer : this->lppInputFromLayer)
	{
		intputBufferCount += layer->GetOutputBufferCount();
	}
	return intputBufferCount;
}


/** 入力元レイヤーへのリンクを追加する.
	@param	pLayer	追加する入力元レイヤー */
ELayerErrorCode NNLayer_FeedforwardBase::AddInputFromLayer(IOutputLayer* pLayer)
{
	// 同じ入力レイヤーが存在しない確認する
	for(auto it : this->lppInputFromLayer)
	{
		if(it == pLayer)
			return LAYER_ERROR_ADDLAYER_ALREADY_SAMEID;
	}


	// リストに追加
	this->lppInputFromLayer.push_back(pLayer);

	// 入力元レイヤーに自分を出力先として追加
	pLayer->AddOutputToLayer(this);

	return LAYER_ERROR_NONE;
}
/** 入力元レイヤーへのリンクを削除する.
	@param	pLayer	削除する出力先レイヤー */
ELayerErrorCode NNLayer_FeedforwardBase::EraseInputFromLayer(IOutputLayer* pLayer)
{
	// リストから検索して削除
	auto it = this->lppInputFromLayer.begin();
	while(it != this->lppInputFromLayer.end())
	{
		if(*it == pLayer)
		{
			// リストから削除
			this->lppInputFromLayer.erase(it);

			// 削除レイヤーに登録されている自分自身を削除
			pLayer->EraseOutputToLayer(this);

			return LAYER_ERROR_NONE;
		}
		it++;
	}

	return LAYER_ERROR_ERASELAYER_NOTFOUND;
}

/** 入力元レイヤー数を取得する */
unsigned int NNLayer_FeedforwardBase::GetInputFromLayerCount()const
{
	return this->lppInputFromLayer.size();
}
/** 入力元レイヤーのアドレスを番号指定で取得する.
	@param num	取得するレイヤーの番号.
	@return	成功した場合入力元レイヤーのアドレス.失敗した場合はNULLが返る. */
IOutputLayer* NNLayer_FeedforwardBase::GetInputFromLayerByNum(unsigned int num)const
{
	if(num >= this->lppInputFromLayer.size())
		return NULL;

	return this->lppInputFromLayer[num];
}

/** 入力元レイヤーが入力バッファのどの位置に居るかを返す.
	※対象入力レイヤーの前にいくつの入力バッファが存在するか.
	　学習差分の使用開始位置としても使用する.
	@return 失敗した場合負の値が返る */
int NNLayer_FeedforwardBase::GetInputBufferPositionByLayer(const IOutputLayer* pLayer)
{
	unsigned int bufferPos = 0;

	for(unsigned int layerNum=0; layerNum<this->lppInputFromLayer.size(); layerNum++)
	{
		if(this->lppInputFromLayer[layerNum] == pLayer)
			return bufferPos;

		bufferPos += this->lppInputFromLayer[layerNum]->GetOutputBufferCount();
	}

	return -1;
}


//===========================
// 出力レイヤー関連
//===========================
/** 出力データ構造を取得する */
IODataStruct NNLayer_FeedforwardBase::GetOutputDataStruct()const
{
	if(this->pConfig == NULL)
		return IODataStruct();

	IODataStruct outputDataStruct;

	outputDataStruct.x = 1;
	outputDataStruct.y = 1;
	outputDataStruct.z = 1;
	outputDataStruct.t = 1;
	outputDataStruct.ch = this->GetNeuronCount();

	return outputDataStruct;
}

/** 出力バッファ数を取得する */
unsigned int NNLayer_FeedforwardBase::GetOutputBufferCount()const
{
	IODataStruct outputDataStruct = GetOutputDataStruct();

	return outputDataStruct.ch * outputDataStruct.x * outputDataStruct.y * outputDataStruct.z * outputDataStruct.t;
}

/** 出力先レイヤーへのリンクを追加する.
	@param	pLayer	追加する出力先レイヤー
	@return	成功した場合0 */
ELayerErrorCode NNLayer_FeedforwardBase::AddOutputToLayer(class IInputLayer* pLayer)
{
	// 同じ出力先レイヤーが存在しない確認する
	for(auto it : this->lppOutputToLayer)
	{
		if(it == pLayer)
			return LAYER_ERROR_ADDLAYER_ALREADY_SAMEID;
	}


	// リストに追加
	this->lppOutputToLayer.push_back(pLayer);

	// 出力先レイヤーに自分を入力元として追加
	pLayer->AddInputFromLayer(this);

	return LAYER_ERROR_NONE;
}
/** 出力先レイヤーへのリンクを削除する.
	@param	pLayer	削除する出力先レイヤー */
ELayerErrorCode NNLayer_FeedforwardBase::EraseOutputToLayer(class IInputLayer* pLayer)
{
	// リストから検索して削除
	auto it = this->lppOutputToLayer.begin();
	while(it != this->lppOutputToLayer.end())
	{
		if(*it == pLayer)
		{
			// リストから削除
			this->lppOutputToLayer.erase(it);

			// 削除レイヤーに登録されている自分自身を削除
			pLayer->EraseInputFromLayer(this);

			return LAYER_ERROR_NONE;
		}
		it++;
	}

	return LAYER_ERROR_ERASELAYER_NOTFOUND;
}

/** 出力先レイヤー数を取得する */
unsigned int NNLayer_FeedforwardBase::GetOutputToLayerCount()const
{
	return this->lppOutputToLayer.size();
}
/** 出力先レイヤーのアドレスを番号指定で取得する.
	@param num	取得するレイヤーの番号.
	@return	成功した場合出力先レイヤーのアドレス.失敗した場合はNULLが返る. */
IInputLayer* NNLayer_FeedforwardBase::GetOutputToLayerByNum(unsigned int num)const
{
	if(num > this->lppOutputToLayer.size())
		return NULL;

	return this->lppOutputToLayer[num];
}

//===========================
// 固有関数
//===========================
/** ニューロン数を取得する */
unsigned int NNLayer_FeedforwardBase::GetNeuronCount()const
{
	if(pConfig == NULL)
		return 0;

	const INNLayerConfigItem_Int* pConfigItem = (const INNLayerConfigItem_Int*)pConfig->GetItemByNum(0);
	if(pConfigItem == NULL)
		return 0;

	return (unsigned int)pConfigItem->GetValue();
}