//======================================
// バッチ正規化レイヤー
//======================================
#include"stdafx.h"

#include"BatchNormalization_FUNC.hpp"

#include"BatchNormalization_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
BatchNormalization_Base::BatchNormalization_Base(Gravisbell::GUID guid)
	:	INNLayer()
	,	guid				(guid)
	,	pLearnData			(NULL)
{
}

/** デストラクタ */
BatchNormalization_Base::~BatchNormalization_Base()
{
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// レイヤー共通
//===========================

/** レイヤー種別の取得.
	ELayerKind の組み合わせ. */
U32 BatchNormalization_Base::GetLayerKindBase(void)const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
}

/** レイヤー固有のGUIDを取得する */
Gravisbell::GUID BatchNormalization_Base::GetGUID(void)const
{
	return this->guid;
}

/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
Gravisbell::GUID BatchNormalization_Base::GetLayerCode(void)const
{
	return this->GetLayerData().GetLayerCode();
}

/** バッチサイズを取得する.
	@return 同時に演算を行うバッチのサイズ */
unsigned int BatchNormalization_Base::GetBatchSize()const
{
	return this->batchSize;
}

/** レイヤーの設定情報を取得する */
const SettingData::Standard::IData* BatchNormalization_Base::GetLayerStructure()const
{
	return this->GetLayerData().GetLayerStructure();
}


//===========================
// 入力レイヤー関連
//===========================
/** 入力データ構造を取得する.
	@return	入力データ構造 */
IODataStruct BatchNormalization_Base::GetInputDataStruct()const
{
	return this->GetLayerData().GetInputDataStruct();
}

/** 入力バッファ数を取得する. */
unsigned int BatchNormalization_Base::GetInputBufferCount()const
{
	return this->GetLayerData().GetInputBufferCount();
}


//===========================
// レイヤー保存
//===========================
/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
unsigned int BatchNormalization_Base::GetUseBufferByteCount()const
{
	return this->GetLayerData().GetUseBufferByteCount();
}


//===========================
// 出力レイヤー関連
//===========================
/** 出力データ構造を取得する */
IODataStruct BatchNormalization_Base::GetOutputDataStruct()const
{
	return this->GetLayerData().GetOutputDataStruct();
}

/** 出力バッファ数を取得する */
unsigned int BatchNormalization_Base::GetOutputBufferCount()const
{
	return this->GetLayerData().GetOutputBufferCount();
}


//===========================
// 固有関数
//===========================
