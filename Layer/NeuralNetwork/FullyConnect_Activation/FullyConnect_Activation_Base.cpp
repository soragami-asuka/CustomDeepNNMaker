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
	,	guid				(guid)
	,	pLearnData			(NULL)
{
}

/** デストラクタ */
FullyConnect_Activation_Base::~FullyConnect_Activation_Base()
{
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// レイヤー共通
//===========================

/** レイヤー種別の取得.
	ELayerKind の組み合わせ. */
U32 FullyConnect_Activation_Base::GetLayerKindBase(void)const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
}

/** レイヤー固有のGUIDを取得する */
Gravisbell::GUID FullyConnect_Activation_Base::GetGUID(void)const
{
	return this->guid;
}

/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
Gravisbell::GUID FullyConnect_Activation_Base::GetLayerCode(void)const
{
	return this->GetLayerData().GetLayerCode();
}

/** バッチサイズを取得する.
	@return 同時に演算を行うバッチのサイズ */
unsigned int FullyConnect_Activation_Base::GetBatchSize()const
{
	return this->batchSize;
}

/** レイヤーの設定情報を取得する */
const SettingData::Standard::IData* FullyConnect_Activation_Base::GetLayerStructure()const
{
	return this->GetLayerData().GetLayerStructure();
}


//===========================
// 入力レイヤー関連
//===========================
/** 入力データ構造を取得する.
	@return	入力データ構造 */
IODataStruct FullyConnect_Activation_Base::GetInputDataStruct()const
{
	return this->GetLayerData().GetInputDataStruct();
}

/** 入力バッファ数を取得する. */
unsigned int FullyConnect_Activation_Base::GetInputBufferCount()const
{
	return this->GetLayerData().GetInputBufferCount();
}


//===========================
// レイヤー保存
//===========================
/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
unsigned int FullyConnect_Activation_Base::GetUseBufferByteCount()const
{
	return this->GetLayerData().GetUseBufferByteCount();
}


//===========================
// 出力レイヤー関連
//===========================
/** 出力データ構造を取得する */
IODataStruct FullyConnect_Activation_Base::GetOutputDataStruct()const
{
	return this->GetLayerData().GetOutputDataStruct();
}

/** 出力バッファ数を取得する */
unsigned int FullyConnect_Activation_Base::GetOutputBufferCount()const
{
	return this->GetLayerData().GetOutputBufferCount();
}


//===========================
// 固有関数
//===========================
/** ニューロン数を取得する */
U32 FullyConnect_Activation_Base::GetNeuronCount()const
{
	return this->GetLayerData().GetNeuronCount();
}
