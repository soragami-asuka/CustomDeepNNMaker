//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward_FUNC.hpp"

#include"NNLayer_FeedforwardBase.h"

using namespace Gravisbell;
using namespace Gravisbell::NeuralNetwork;


/** コンストラクタ */
NNLayer_FeedforwardBase::NNLayer_FeedforwardBase(Gravisbell::GUID guid)
	:	INNLayer()
	,	guid			(guid)
	,	pLayerStructure	(NULL)
	,	pLearnData		(NULL)
	,	lppInputFromLayer	()		/**< 入力元レイヤーのリスト */
	,	lppOutputToLayer	()		/**< 出力先レイヤーのリスト */
{
}

/** デストラクタ */
NNLayer_FeedforwardBase::~NNLayer_FeedforwardBase()
{
	if(pLayerStructure != NULL)
		delete pLayerStructure;
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// レイヤー共通
//===========================

/** レイヤー種別の取得.
	ELayerKind の組み合わせ. */
unsigned int NNLayer_FeedforwardBase::GetLayerKindBase()const
{
	return Gravisbell::NeuralNetwork::LAYER_KIND_CALC | Gravisbell::NeuralNetwork::LAYER_KIND_SINGLE_INPUT | Gravisbell::NeuralNetwork::LAYER_KIND_SINGLE_OUTPUT;
}

/** レイヤー固有のGUIDを取得する */
Gravisbell::ErrorCode NNLayer_FeedforwardBase::GetGUID(Gravisbell::GUID& o_guid)const
{
	o_guid = this->guid;

	return ERROR_CODE_NONE;
}

/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
Gravisbell::ErrorCode NNLayer_FeedforwardBase::GetLayerCode(Gravisbell::GUID& o_layerCode)const
{
	return ::GetLayerCode(o_layerCode);
}

/** バッチサイズを取得する.
	@return 同時に演算を行うバッチのサイズ */
unsigned int NNLayer_FeedforwardBase::GetBatchSize()const
{
	return this->batchSize;
}


/** 設定情報を設定 */
Gravisbell::ErrorCode NNLayer_FeedforwardBase::SetLayerConfig(const SettingData::Standard::IData& config)
{
	Gravisbell::ErrorCode err = ERROR_CODE_NONE;

	// レイヤーコードを確認
	{
		GUID config_guid;
		err = config.GetLayerCode(config_guid);
		if(err != ERROR_CODE_NONE)
			return err;

		GUID layer_guid;
		err = ::GetLayerCode(layer_guid);
		if(err != ERROR_CODE_NONE)
			return err;

		if(config_guid != layer_guid)
			return ERROR_CODE_INITLAYER_DISAGREE_CONFIG;
	}

	if(this->pLayerStructure != NULL)
		delete this->pLayerStructure;
	this->pLayerStructure = config.Clone();

	// 構造体に読み込む
	this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

	return ERROR_CODE_NONE;
}
/** レイヤーの設定情報を取得する */
const SettingData::Standard::IData* NNLayer_FeedforwardBase::GetLayerConfig()const
{
	return this->pLayerStructure;
}


//===========================
// 入力レイヤー関連
//===========================
/** 入力データ構造を取得する.
	@return	入力データ構造 */
IODataStruct NNLayer_FeedforwardBase::GetInputDataStruct()const
{
	return this->inputDataStruct;
}

/** 入力バッファ数を取得する. */
unsigned int NNLayer_FeedforwardBase::GetInputBufferCount()const
{
	return this->inputDataStruct.x * this->inputDataStruct.y * this->inputDataStruct.z * this->inputDataStruct.ch;
}


//===========================
// レイヤー保存
//===========================
/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
unsigned int NNLayer_FeedforwardBase::GetUseBufferByteCount()const
{
	unsigned int bufferSize = 0;

	if(pLayerStructure == NULL)
		return 0;

	// 設定情報
	bufferSize += pLayerStructure->GetUseBufferByteCount();

	// 本体のバイト数
	bufferSize += (this->GetNeuronCount() * this->GetInputBufferCount()) * sizeof(NEURON_TYPE);	// ニューロン係数
	bufferSize += this->GetNeuronCount() * sizeof(NEURON_TYPE);	// バイアス係数


	return bufferSize;
}


//===========================
// 出力レイヤー関連
//===========================
/** 出力データ構造を取得する */
IODataStruct NNLayer_FeedforwardBase::GetOutputDataStruct()const
{
	IODataStruct outputDataStruct;

	outputDataStruct.x = 1;
	outputDataStruct.y = 1;
	outputDataStruct.z = 1;
	outputDataStruct.ch = this->GetNeuronCount();

	return outputDataStruct;
}

/** 出力バッファ数を取得する */
unsigned int NNLayer_FeedforwardBase::GetOutputBufferCount()const
{
	IODataStruct outputDataStruct = GetOutputDataStruct();

	return outputDataStruct.x * outputDataStruct.y * outputDataStruct.z * outputDataStruct.ch;
}


//===========================
// 固有関数
//===========================
/** ニューロン数を取得する */
unsigned int NNLayer_FeedforwardBase::GetNeuronCount()const
{
	return this->layerStructure.NeuronCount;
}