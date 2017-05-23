//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"Residual_FUNC.hpp"

#include"Residual_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
Residual_Base::Residual_Base(Gravisbell::GUID guid)
	:	guid				(guid)
	,	pLearnData			(NULL)
{
}

/** デストラクタ */
Residual_Base::~Residual_Base()
{
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// レイヤー共通
//===========================

/** レイヤー種別の取得.
	ELayerKind の組み合わせ. */
U32 Residual_Base::GetLayerKindBase(void)const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
}

/** レイヤー固有のGUIDを取得する */
Gravisbell::GUID Residual_Base::GetGUID(void)const
{
	return this->guid;
}

/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
Gravisbell::GUID Residual_Base::GetLayerCode(void)const
{
	return this->GetLayerData().GetLayerCode();
}

/** バッチサイズを取得する.
	@return 同時に演算を行うバッチのサイズ */
unsigned int Residual_Base::GetBatchSize()const
{
	return this->batchSize;
}

/** レイヤーの設定情報を取得する */
const SettingData::Standard::IData* Residual_Base::GetLayerStructure()const
{
	return this->GetLayerData().GetLayerStructure();
}


//===========================
// 入力レイヤー関連
//===========================
/** 入力データの数を取得する */
U32 Residual_Base::GetInputDataCount()const
{
	return this->GetLayerData().GetInputDataCount();
}

/** 入力データ構造を取得する.
	@return	入力データ構造 */
IODataStruct Residual_Base::GetInputDataStruct(U32 i_dataNum)const
{
	return this->GetLayerData().GetInputDataStruct(i_dataNum);
}

/** 入力バッファ数を取得する. */
unsigned int Residual_Base::GetInputBufferCount(U32 i_dataNum)const
{
	return this->GetLayerData().GetInputBufferCount(i_dataNum);
}


//===========================
// 出力レイヤー関連
//===========================
/** 出力データ構造を取得する */
IODataStruct Residual_Base::GetOutputDataStruct()const
{
	return this->GetLayerData().GetOutputDataStruct();
}

/** 出力バッファ数を取得する */
unsigned int Residual_Base::GetOutputBufferCount()const
{
	return this->GetLayerData().GetOutputBufferCount();
}


//===========================
// 固有関数
//===========================
