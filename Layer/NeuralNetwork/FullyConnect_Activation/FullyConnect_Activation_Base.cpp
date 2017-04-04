//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#include"stdafx.h"

#include"FullyConnect_Activation_FUNC.hpp"

#include"FullyConnect_Activation_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
FullyConnect_Activation_Base::FullyConnect_Activation_Base(Gravisbell::GUID guid)
	:	INNLayer()
	,	guid			(guid)
	,	pLayerStructure	(NULL)
	,	pLearnData		(NULL)
{
}

/** デストラクタ */
FullyConnect_Activation_Base::~FullyConnect_Activation_Base()
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
unsigned int FullyConnect_Activation_Base::GetLayerKindBase()const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
}

/** レイヤー固有のGUIDを取得する */
Gravisbell::ErrorCode FullyConnect_Activation_Base::GetGUID(Gravisbell::GUID& o_guid)const
{
	o_guid = this->guid;

	return ERROR_CODE_NONE;
}

/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
Gravisbell::ErrorCode FullyConnect_Activation_Base::GetLayerCode(Gravisbell::GUID& o_layerCode)const
{
	return ::GetLayerCode(o_layerCode);
}

/** バッチサイズを取得する.
	@return 同時に演算を行うバッチのサイズ */
unsigned int FullyConnect_Activation_Base::GetBatchSize()const
{
	return this->batchSize;
}


/** 設定情報を設定 */
Gravisbell::ErrorCode FullyConnect_Activation_Base::SetLayerConfig(const SettingData::Standard::IData& config)
{
	Gravisbell::ErrorCode err = ERROR_CODE_NONE;

	// レイヤーコードを確認
	{
		Gravisbell::GUID config_guid;
		err = config.GetLayerCode(config_guid);
		if(err != ERROR_CODE_NONE)
			return err;

		Gravisbell::GUID layer_guid;
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
const SettingData::Standard::IData* FullyConnect_Activation_Base::GetLayerConfig()const
{
	return this->pLayerStructure;
}


//===========================
// 入力レイヤー関連
//===========================
/** 入力データ構造を取得する.
	@return	入力データ構造 */
IODataStruct FullyConnect_Activation_Base::GetInputDataStruct()const
{
	return this->inputDataStruct;
}

/** 入力バッファ数を取得する. */
unsigned int FullyConnect_Activation_Base::GetInputBufferCount()const
{
	return this->inputDataStruct.x * this->inputDataStruct.y * this->inputDataStruct.z * this->inputDataStruct.ch;
}


//===========================
// レイヤー保存
//===========================
/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
unsigned int FullyConnect_Activation_Base::GetUseBufferByteCount()const
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
IODataStruct FullyConnect_Activation_Base::GetOutputDataStruct()const
{
	IODataStruct outputDataStruct;

	outputDataStruct.x = 1;
	outputDataStruct.y = 1;
	outputDataStruct.z = 1;
	outputDataStruct.ch = this->GetNeuronCount();

	return outputDataStruct;
}

/** 出力バッファ数を取得する */
unsigned int FullyConnect_Activation_Base::GetOutputBufferCount()const
{
	IODataStruct outputDataStruct = GetOutputDataStruct();

	return outputDataStruct.x * outputDataStruct.y * outputDataStruct.z * outputDataStruct.ch;
}


//===========================
// 固有関数
//===========================
/** ニューロン数を取得する */
unsigned int FullyConnect_Activation_Base::GetNeuronCount()const
{
	return this->layerStructure.NeuronCount;
}